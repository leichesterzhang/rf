from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
d = Document('ref.docx')
def font(st, size, east, bold=False, west='Times New Roman'):
    st.font.name = west; st.font.size = Pt(size); st.font.bold = bold
    st.font.italic = False; st.font.color.rgb = RGBColor(0,0,0)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = rpr.makeelement(qn('w:rFonts'), {}); rpr.append(rf)
    for k in ('ascii','hAnsi','cs'): rf.set(qn('w:'+k), west)
    rf.set(qn('w:eastAsia'), east)
    for k in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
        rf.attrib.pop(qn('w:'+k), None)
names = [s.name for s in d.styles]
S = {s.name: s for s in d.styles}
for n in ('Normal','Body Text','First Paragraph'):
    st = S[n]; font(st, 12, '宋体')
    pf = st.paragraph_format; pf.line_spacing = 1.5; pf.first_line_indent = Pt(24)
    pf.space_before = Pt(0); pf.space_after = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for n, sz in (('Heading 1',16),('Heading 2',14),('Heading 3',12),('Title',18)):
    st = S[n]; font(st, sz, '黑体', bold=True)
    pf = st.paragraph_format; pf.first_line_indent = Pt(0)
    pf.space_before = Pt(12); pf.space_after = Pt(6); pf.line_spacing = 1.5
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
for n in ('Image Caption','Table Caption','Caption'):
    if n in names:
        st = S[n]; font(st, 10.5, '宋体')
        pf = st.paragraph_format; pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0); pf.space_before = Pt(3); pf.space_after = Pt(9); pf.line_spacing=1.2
for n in ('Captioned Figure','Figure'):
    if n in names:
        st = S[n]; pf = st.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER; pf.first_line_indent = Pt(0); pf.space_before = Pt(6)
if 'Compact' in names:
    st = S["Compact"]; font(st, 10.5, '宋体')
    pf = st.paragraph_format; pf.first_line_indent = Pt(0); pf.line_spacing = 1.15
    pf.space_before = Pt(1); pf.space_after = Pt(1)
if 'Bibliography' in names: pass
sec = d.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5); sec.top_margin = sec.bottom_margin = Cm(2.54)
d.save('ref.docx')
