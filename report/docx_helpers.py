"""Small helpers around python-docx for an academic-style report."""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

BODY_FONT = "Cambria"
HEAD_FONT = "Calibri"
ACCENT = RGBColor(0x9E, 0x2A, 0x2B)
DARK = RGBColor(0x1F, 0x2A, 0x36)


class Report:
    def __init__(self):
        self.doc = Document()
        self.fig_n = 0
        self.tab_n = 0
        self.headings = []
        self.toc_marker = None
        s = self.doc.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin"):
            setattr(s, side, Cm(2.4))
        s.top_margin = s.bottom_margin = Cm(2.3)
        st = self.doc.styles
        normal = st["Normal"]
        normal.font.name = BODY_FONT
        normal.font.size = Pt(11)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        pf = normal.paragraph_format
        pf.space_after = Pt(6)
        pf.line_spacing = 1.15
        for lvl, size, before in [(1, 16, 18), (2, 13, 12), (3, 11.5, 10)]:
            h = st[f"Heading {lvl}"]
            h.font.name = HEAD_FONT
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = DARK if lvl > 1 else ACCENT
            rf = h.element.rPr.rFonts
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), HEAD_FONT)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            h.paragraph_format.space_before = Pt(before)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
        cap = st["Caption"]
        cap.font.name = HEAD_FONT
        cap.font.size = Pt(9)
        cap.font.italic = False
        cap.font.bold = False
        cap.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        self._footer_page_numbers()

    # ------------------------------------------------------------------
    def _footer_page_numbers(self):
        p = self.doc.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._field(p, "PAGE")

    @staticmethod
    def _field(paragraph, code):
        r = paragraph.add_run()
        for kind, text in (("begin", None), ("instr", code), ("separate", None),
                           ("text", "1"), ("end", None)):
            if kind == "instr":
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = f" {text} "
            elif kind == "text":
                el = OxmlElement("w:t")
                el.text = text
            else:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind)
            r._r.append(el)
        r.font.size = Pt(9)

    def toc(self):
        """Mark where the table of contents goes; filled by fill_toc()."""
        self.toc_marker = self.doc.add_paragraph()

    def fill_toc(self, pages=None, levels=(1, 2)):
        """Static TOC: one line per heading with a dot leader and page number.

        `pages` maps heading text -> page number. On the first build pass it
        is None and '00' is written, so the TOC has the same length in both
        passes and the pagination does not move.
        """
        from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
        sec = self.doc.sections[0]
        right = sec.page_width - sec.left_margin - sec.right_margin
        for lvl, text in self.headings:
            if lvl not in levels or text == "Contents":
                continue
            para = self.toc_marker.insert_paragraph_before()
            pf = para.paragraph_format
            pf.tab_stops.add_tab_stop(right, WD_TAB_ALIGNMENT.RIGHT,
                                      WD_TAB_LEADER.DOTS)
            pf.left_indent = Cm(0 if lvl == 1 else 0.8)
            pf.space_after = Pt(1 if lvl == 2 else 2)
            pf.space_before = Pt(5 if lvl == 1 else 0)
            pf.line_spacing = 1.0
            num = "00" if pages is None else str(pages.get(text, ""))
            r = para.add_run(f"{text}\t{num}")
            r.font.size = Pt(10.5 if lvl == 1 else 10)
            r.bold = lvl == 1
        # drop the empty marker so it cannot spill onto a page of its own
        self.toc_marker._p.getparent().remove(self.toc_marker._p)

    # ------------------------------------------------------------------
    def h(self, text, level=1):
        self.headings.append((level, text))
        return self.doc.add_heading(text, level=level)

    def p(self, *parts, align=None, style=None, space_after=None, size=None):
        """Paragraph from strings; '**x**' -> bold, '*x*' -> italic."""
        para = self.doc.add_paragraph(style=style)
        for part in parts:
            self._runs(para, part, size)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "justify":
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if space_after is not None:
            para.paragraph_format.space_after = Pt(space_after)
        return para

    @staticmethod
    def _runs(para, text, size=None, bold=False):
        """'**x**' -> bold, '*x*' -> italic, '`x`' -> monospace."""
        import re
        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = para.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("`"):
                r = para.add_run(tok[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt((size or 11) - 1.5)
                r.bold = bold or None
                continue
            elif tok.startswith("*"):
                r = para.add_run(tok[1:-1]); r.italic = True
            else:
                r = para.add_run(tok)
            if bold:
                r.bold = True
            if size:
                r.font.size = Pt(size)

    def bullets(self, items, numbered=False, size=None):
        style = "List Number" if numbered else "List Bullet"
        for it in items:
            para = self.doc.add_paragraph(style=style)
            self._runs(para, it, size)
            para.paragraph_format.space_after = Pt(3)

    def eq(self, text):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(text)
        r.font.name = "Cambria Math"
        r.italic = True
        return para

    def figure(self, path, caption, width_cm=15.5):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        self.fig_n += 1
        c = self.doc.add_paragraph(style="Caption")
        r = c.add_run(f"Figure {self.fig_n}. "); r.bold = True
        self._runs(c, caption)
        return self.fig_n

    def table(self, header, rows, caption, widths_cm=None, font=8.5,
              bold_rows=(), left_cols=(0,)):
        """Table with a dark header row. Columns in `left_cols` are
        left-aligned (text); the others are right-aligned (numbers). The
        header repeats on a new page and small tables are kept on one page."""
        self.tab_n += 1
        c = self.doc.add_paragraph(style="Caption")
        c.paragraph_format.keep_with_next = True
        r = c.add_run(f"Table {self.tab_n}. "); r.bold = True
        self._runs(c, caption)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, hd in enumerate(header):
            cell = t.rows[0].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(hd)
            run.bold = True
            run.font.size = Pt(font)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            self._shade(cell, "1F2A36")
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = ""
                self._runs(cells[j].paragraphs[0], str(v), size=font,
                           bold=i in bold_rows)
                if i in bold_rows:
                    self._shade(cells[j], "F3E3E3")
                if j not in left_cols:
                    cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        # repeat header row; never split a row; keep short tables together
        trPr = t.rows[0]._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true")
        trPr.append(el)
        keep = len(t.rows) <= 25
        for ri, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            el = OxmlElement("w:cantSplit"); el.set(qn("w:val"), "true")
            trPr.append(el)
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(0)
                    para.paragraph_format.line_spacing = 1.0
                    if keep and ri < len(t.rows) - 1:
                        para.paragraph_format.keep_with_next = True
        if widths_cm:
            for row in t.rows:
                for j, w in enumerate(widths_cm):
                    row.cells[j].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return self.tab_n

    @staticmethod
    def _shade(cell, hex_):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_)
        tcPr.append(shd)

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        self.doc.save(str(path))
