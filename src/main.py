# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, Tuple

from rich.live import Live

from src.config import QueryPlanConfig
from src.generator import answer, dedupe_generated_text
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts
from src.query_enhancement import generate_hypothetical_document
from rich.console import Console
from rich.markdown import Markdown

# Citation Manager Imports
from src.citations import CitationManager
from src.generator import answer_with_citations

# Confidence Scoring Imports
from src.generator import get_confidence_after_generation

# Cache Imports
from src.query_cache import QueryCache
from src.embedder import SentenceTransformer

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
        help="system prompt mode (choices: baseline, tutor, concise, detailed)"
    )
    
    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    # Citation Argument for the style user wants
    parser.add_argument(
        "--citation_style",
        choices=["minimal", "detailed", "numbered", "none"],
        default="minimal",
        help="citation format style (default: minimal)"
    )

    # Add cache argument group
    cache_group = parser.add_argument_group("cache options")
    cache_group.add_argument(
        "--enable_cache",
        action="store_true",
        help="enable query caching for faster responses"
    )
    cache_group.add_argument(
        "--cache_threshold",
        type=float,
        default=0.85,
        help="similarity threshold for cache hits (0.0-1.0, default: 0.85)"
    )
    cache_group.add_argument(
        "--clear_cache",
        action="store_true",
        help="clear the cache before starting"
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_with_pages.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
    )

def use_indexed_chunks(question: str, chunks: list, logger: "RunLogger") -> list:
    """
    Retrieve chunks from the indexed chunks based on simple keyword matching.
    """
    with open('index/sections/textbook_index_page_to_chunk_map.json', 'r') as f:
            page_to_chunk_map = json.load(f)
    with open('data/extracted_index.json', 'r') as f:
        extracted_index = json.load(f)

    keywords = get_keywords(question)
    chunk_ids = set()
    ranked_chunks = []

    print(f"Extracted keywords for indexed chunk retrieval: {keywords}")

    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
            
    for cid in chunk_ids:
        ranked_chunks.append(chunks[cid])

    print(f"Chunks retrieved using indexed chunks: {len(ranked_chunks)}")
    return ranked_chunks

def get_answer(
    question: str,
    cfg: QueryPlanConfig,
    args: argparse.Namespace,
    logger: "RunLogger",
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False
) -> Tuple:
    """
    Run a single query through the pipeline.
    """

    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    metadata_list = artifacts.get("metadata", []) # Added this for citations
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    
    logger.log_query_start(question)
    citation_manager = CitationManager()
    
    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    topk_idxs = []  # Initialize to avoid UnboundLocalError
    
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
        topk_idxs = list(range(len(golden_chunks)))  # Create dummy indices
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        # Use chunks from the textbook index
        ranked_chunks = use_indexed_chunks(question, chunks, logger)
        topk_idxs = list(range(len(ranked_chunks)))  # Create dummy indices
    else:
        # Step 0: Query Enhancement (HyDE)
        retrieval_query = question
        if cfg.use_hyde:
            model_path = args.model_path or cfg.model_path
            hypothetical_doc = generate_hypothetical_document(
                question, model_path, max_tokens=cfg.hyde_max_tokens
            )
            retrieval_query = hypothetical_doc
            hyde_query = hypothetical_doc
            # print(f"🔍 HyDE query: {hypothetical_doc}")
        
        # Step 1: Retrieval
        pool_n = max(cfg.pool_size, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
        # TODO: Fix retrieval logging.
        
        # Step 2: Ranking
        ordered = ranker.rank(raw_scores=raw_scores)
        topk_idxs = apply_seg_filter(cfg, chunks, ordered)
        logger.log_chunks_used(topk_idxs, chunks, sources)
        
        ranked_chunks = [chunks[i] for i in topk_idxs]
        
        # Capture chunk info if in test mode
        if is_test_mode:
            # Compute individual ranker ranks
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores = raw_scores.get("bm25", {})
            
            faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)  # Higher score = better
            bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)  # Higher score = better
            
            faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
            
            chunks_info = []
            for rank, idx in enumerate(topk_idxs, 1):
                chunks_info.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "content": chunks[idx],
                    "faiss_score": faiss_scores.get(idx, 0),
                    "faiss_rank": faiss_ranks.get(idx, 0),
                    "bm25_score": bm25_scores.get(idx, 0),
                    "bm25_rank": bm25_ranks.get(idx, 0),
                })
        
        # Step 3: Final Re-ranking (if enabled)
        # Disabled till we fix the core pipeline
        # ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.top_k)

    # Prepare metadata for citations
    chunk_metadata = []
    for idx in topk_idxs:
        meta = metadata_list[idx] if idx < len(metadata_list) else {}
        chunk_metadata.append(meta)
        # Track in citation manager
        citation_manager.add_chunk(
            chunk_id=idx,
            content=chunks[idx],
            metadata=meta
        )
    
    # Step 4: Generation with Citations
    model_path = args.model_path or cfg.model_path
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode

    stream_iter, citation_manager = answer_with_citations(
        question,
        ranked_chunks,
        chunk_metadata,
        model_path,
        max_tokens=cfg.max_gen_tokens,
        system_prompt_mode=system_prompt,
    )

    if is_test_mode:
        # We do not render MD in the test mode
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        
        # Add citations
        citation_style = getattr(args, 'citation_style', 'minimal')
        if citation_style != 'none':
            citations = citation_manager.format_citations(citation_style) # defaults to minimal if not specified
            ans += citations
        
        # Get confidence score
        confidence = get_confidence_after_generation(ans, question, ranked_chunks, model_path)
        
        return ans, chunks_info, hyde_query, citation_manager, confidence

        return ans, chunks_info, hyde_query, citation_manager
    else:
        # Accumulate the full text while rendering incremental Markdown chunks
        ans = render_streaming_ans(console, stream_iter)
        
        # Add citations after answer
        citation_style = getattr(args, 'citation_style', 'minimal')
        if citation_style != 'none':
            citations = citation_manager.format_citations(citation_style)
            if console:
                console.print(Markdown(citations))
            ans += citations
        
        # Get confidence score after generation
            print("\n[Confidence] Calculating confidence score...")
            confidence = get_confidence_after_generation(ans, question, ranked_chunks, model_path)
        
        # Display confidence with color coding
        if confidence >= 80:
            color = "green"
        elif confidence >= 60:
            color = "yellow"
        else:
            color = "red"

        console.print(f"\n[{color}] Confidence: {confidence}%[/{color}]\n")

        # Return consistent tuple for normal mode
        return ans, citation_manager, confidence


def render_streaming_ans(console, stream_iter):
    if not console:
        raise ValueError("Console must be non null for rendering.")
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=5) as live:
        for delta in stream_iter:
            if is_first:
                # we need to do this to ensure this marker comes after warning noise in Macs.
                console.print("\n[bold cyan]==================== START OF ANSWER ===================[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]===================== END OF ANSWER ====================[/bold cyan]\n")
    return ans

def get_keywords(question: str) -> list:
    """
    Simple keyword extraction from the question.
    """
    stopwords = set([
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what"
    ])
    words = question.lower().split()
    keywords = [word.strip('.,!?()[]') for word in words if word not in stopwords]
    return keywords

def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()
    console = Console()

    # planner = HeuristicQueryPlanner(cfg)

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")

    # Initializing the cache
    cache = None
    cache_embedder = None
    if args.enable_cache:
        cache = QueryCache(
            cache_dir="cache",
            similarity_threshold=args.cache_threshold,
            max_cache_size=100
        )
        
        if args.clear_cache:
            cache.clear()
            print("[Cache] Cache cleared as requested")
        
        cache_embedder = SentenceTransformer(cfg.embed_model)
        print(f"[Cache] Caching enabled with threshold {args.cache_threshold}")
        
        # Print cache stats if cache exists
        stats = cache.get_stats()
        if stats['size'] > 0:
            print(f"[Cache] Loaded cache with {stats['size']} entries")

    try:
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix=args.index_prefix
        )
        
        # Load metadata for citations
        import pickle
        metadata_path = artifacts_dir / f"{args.index_prefix}_meta.pkl"
        metadata = []
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )
        
        # Package artifacts with metadata
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "metadata": metadata,
            "retrievers": retrievers,
            "ranker": ranker
        }
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print(f"Citation style: {args.citation_style}")
    if args.enable_cache:
        print(f"Cache: ENABLED (threshold: {args.cache_threshold})")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'cache stats' to view cache statistics.\n")
    
    while True:
        try:
            q = input("Ask > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                if cache:
                    stats = cache.get_stats()
                    print(f"\n[Cache] Session stats: {stats['size']} cached queries, {stats['total_accesses']} total accesses")
                print("Goodbye!")
                break
            
            # Special command to view cache stats
            if q.lower() == "cache stats" and cache:
                stats = cache.get_stats()
                console.print("\n[bold cyan]Cache Statistics:[/bold cyan]")
                console.print(f"  Size: {stats['size']} queries")
                console.print(f"  Total accesses: {stats['total_accesses']}")
                console.print(f"  Avg accesses per query: {stats['avg_accesses_per_query']:.2f}")
                console.print(f"  Similarity threshold: {stats['similarity_threshold']}")
                console.print(f"  Cache directory: {stats['cache_dir']}")
                continue

            # Check cache if enabled
            cached_result = None
            query_embedding = None
            use_cached = False

            if cache and cache_embedder:
                # Compute query embedding for cache lookup
                query_embedding = cache_embedder.encode([q])[0]
                cached_result = cache.get(q, query_embedding)
                
                if cached_result:
                    # Display cached answer with full details
                    console.print("\n[bold yellow]Serving from Cache[/bold yellow]")
                    console.print(f"[dim]Original query: {cached_result['original_query']}[/dim]")
                    console.print(f"[dim]Similarity: {cached_result['similarity_score']:.3f}[/dim]")
                    console.print(f"[dim]Cached on: {cached_result['timestamp']}[/dim]\n")

                    # Display confidence if available
                    if cached_result.get('confidence') is not None:
                        conf = cached_result['confidence']
                        if conf >= 80:
                            console.print(f"[dim green]Confidence: {conf}%[/dim green]")
                        elif conf >= 60:
                            console.print(f"[dim yellow]Confidence: {conf}%[/dim yellow]")
                        else:
                            console.print(f"[dim red]Confidence: {conf}%[/dim red]")
                    console.print()
                    
                    # Display the cached answer
                    console.print("[bold cyan]==================== CACHED ANSWER ===================[/bold cyan]\n")
                    console.print(Markdown(cached_result['answer']))
                    console.print("\n[bold cyan]===================== END OF ANSWER ====================[/bold cyan]\n")
                    
                    # Ask user for confirmation
                    refetch = input("Use cached answer? (Y/n): ").strip().lower()
                    
                    if refetch in {'y', 'yes', ''}:
                        use_cached = True
                        print("[Cache] Using cached result\n")
                        # Continue to next iteration - don't generate new answer
                        continue
                    else:
                        print("[Cache] Refetching fresh answer...\n")

            # Generate new answer only if not using cached result
            ans, citation_manager, confidence = get_answer(
                q, cfg, args, logger, console, artifacts=artifacts, is_test_mode=False
            )
            
            logger.log_generation(
                ans, 
                {
                    "max_tokens": cfg.max_gen_tokens, 
                    "model_path": args.model_path or cfg.model_path,
                }
            )

            # Add to cache if enabled and not a cache hit
            if cache and cache_embedder and query_embedding is not None:
                cache.add(
                    query=q,
                    query_embedding=query_embedding,
                    answer=ans,
                    citation_manager=citation_manager,
                    confidence=confidence  # Add confidence to cache
                )

        except KeyboardInterrupt:
            if cache:
                stats = cache.get_stats()
                print(f"\n[Cache] Session stats: {stats['size']} cached queries, {stats['total_accesses']} total accesses")
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(str(e))
            break

    # TODO: Fix completion logging.
    # logger.log_query_complete()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Config loading
    config_path = pathlib.Path("config/config.yaml")
    cfg = None
    if config_path.exists():
        cfg = QueryPlanConfig.from_yaml(config_path)

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
