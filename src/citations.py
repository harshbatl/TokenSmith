from typing import List, Dict, Set, Optional
import re


class CitationManager:    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.used_chunks: List[Dict] = []
        self.unique_sources: Set[str] = set()
    
    def add_chunk(self, chunk_id: int, content: str, metadata: Optional[Dict] = None):
        chunk_info = {
            'chunk_id': chunk_id,
            'content': content,
            'metadata': metadata or {}
        }
        self.used_chunks.append(chunk_info)
        
        # Extract source identifier
        if metadata and 'section' in metadata:
            self.unique_sources.add(metadata['section'])
    
    def format_citations(self, style: str = "minimal") -> str:
        # Format the citations based on the styled selected
        # Default is minimal

        if not self.used_chunks:
            return ""
        
        if style == "minimal":
            return self._format_minimal()
        elif style == "detailed":
            return self._format_detailed()
        elif style == "numbered":
            return self._format_numbered()
        else:
            return self._format_minimal()
    
    def _format_minimal(self) -> str:
        # Minimal citation format showing unique sections.
        # Example: "Sources: Section 2.1, Section 3.4"

        if not self.unique_sources:
            return ""
        
        sections = sorted(self.unique_sources)
        return f"\n\n**Sources:** {', '.join(sections)}"
    
    def _format_detailed(self) -> str:
        # Detailed citation format with page numbers and sections like so
        # Sources:
        # - Section 2.1 (Page 45)
        # - Section 3.4 (Pages 67-68)

        if not self.used_chunks:
            return ""
        
        # Group by section
        section_pages: Dict[str, Set[int]] = {}
        for chunk in self.used_chunks:
            meta = chunk['metadata']
            section = meta.get('section', 'Unknown Section')
            page = meta.get('page_number')
            
            if section not in section_pages:
                section_pages[section] = set()
            if page:
                section_pages[section].add(page)
        
        # Format output
        lines = ["\n\n**Sources:**"]
        for section in sorted(section_pages.keys()):
            pages = sorted(section_pages[section])
            if pages:
                page_str = self._format_page_range(pages)
                lines.append(f"- {section} (Page{'s' if len(pages) > 1 else ''} {page_str})")
            else:
                lines.append(f"- {section}")
        
        return "\n".join(lines)
    
    def _format_numbered(self) -> str:
        # Format using a numbered citation format like so:
        # Sources:
        # [1] Section 2.1
        # [2] Section 3.4

        if not self.used_chunks:
            return ""
        
        lines = ["\n\n**Sources:**"]
        for i, chunk in enumerate(self.used_chunks, 1):
            meta = chunk['metadata']
            section = meta.get('section', 'Unknown Section')
            preview = chunk['content'][:80].replace('\n', ' ').strip()
            if len(chunk['content']) > 80:
                preview += "..."
            lines.append(f"[{i}] {section} - \"{preview}\"")
        
        return "\n".join(lines)
    
    def _format_page_range(self, pages: List[int]) -> str:
        # Format a list of pages into a range
        if not pages:
            return ""
        
        pages = sorted(pages)
        ranges = []
        start = pages[0]
        end = pages[0]
        
        for page in pages[1:]:
            if page == end + 1:
                end = page
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = page
        
        # Add final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ", ".join(ranges)
