from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


SLIDE_W = 12_192_000
SLIDE_H = 6_858_000

BG = "F7F3EA"
BG_ALT = "FCF8F1"
TEXT = "1F2937"
MUTED = "5B6470"
ACCENT = "C65D2C"
ACCENT_2 = "0F766E"
ACCENT_3 = "1D4ED8"
ACCENT_4 = "B45309"
PANEL = "FFFDF8"
PANEL_ALT = "F2E7D8"
PANEL_TEAL = "E8F5F2"
PANEL_BLUE = "EAF0FB"
PANEL_SAND = "F8E9D8"
WATERMARK = "E7DDCE"
WHITE = "FFFFFF"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@dataclass
class ShapeCounter:
    next_id: int = 2

    def alloc(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value


def paragraph_xml(
    line: str,
    *,
    size: int,
    color: str,
    bold: bool = False,
    align: str = "l",
) -> str:
    algn_map = {"l": "l", "ctr": "ctr", "r": "r", "just": "just"}
    ppr = f'<a:pPr algn="{algn_map.get(align, "l")}"/>'
    bold_attr = ' b="1"' if bold else ""
    return (
        f"<a:p>{ppr}"
        f'<a:r><a:rPr lang="zh-CN" sz="{size}"{bold_attr}>'
        f"<a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill>"
        f"</a:rPr><a:t>{esc(line)}</a:t></a:r>"
        f'<a:endParaRPr lang="zh-CN" sz="{size}"/></a:p>'
    )


def tx_body(
    lines: Iterable[str],
    *,
    size: int,
    color: str,
    bold: bool = False,
    align: str = "l",
    anchor: str = "t",
) -> str:
    body = "".join(
        paragraph_xml(line, size=size, color=color, bold=bold, align=align)
        for line in lines
    )
    return (
        '<p:txBody>'
        f'<a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720" anchor="{anchor}"/>'
        "<a:lstStyle/>"
        f"{body}"
        "</p:txBody>"
    )


def shape_xml(
    shape_id: int,
    name: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
    *,
    lines: Iterable[str],
    size: int = 1900,
    color: str = TEXT,
    fill: str | None = None,
    line: str | None = None,
    bold: bool = False,
    align: str = "l",
    anchor: str = "t",
    rounded: bool = True,
    txbox: bool = True,
) -> str:
    fill_xml = (
        f"<a:solidFill><a:srgbClr val=\"{fill}\"/></a:solidFill>"
        if fill
        else "<a:noFill/>"
    )
    line_xml = (
        f"<a:ln w=\"12700\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln>"
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    txbox_attr = ' txBox="1"' if txbox else ""
    geom = "roundRect" if rounded else "rect"
    return (
        "<p:sp>"
        "<p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{esc(name)}"/>'
        f"<p:cNvSpPr{txbox_attr}/>"
        "<p:nvPr/>"
        "</p:nvSpPr>"
        "<p:spPr>"
        "<a:xfrm>"
        f'<a:off x="{x}" y="{y}"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/>'
        "</a:xfrm>"
        f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
        f"{fill_xml}"
        f"{line_xml}"
        "</p:spPr>"
        f"{tx_body(lines, size=size, color=color, bold=bold, align=align, anchor=anchor)}"
        "</p:sp>"
    )


@dataclass
class SlideBuilder:
    number: str
    kicker: str
    title: str
    bg: str = BG
    shapes: list[str] = field(default_factory=list)
    counter: ShapeCounter = field(default_factory=ShapeCounter)

    def __post_init__(self) -> None:
        self.add_watermark()
        self.add_kicker()
        self.add_title()

    def add_watermark(self) -> None:
        sid = self.counter.alloc()
        self.shapes.append(
            shape_xml(
                sid,
                "watermark",
                10_050_000,
                120_000,
                1_700_000,
                1_100_000,
                lines=[self.number],
                size=18_000,
                color=WATERMARK,
                align="ctr",
                anchor="ctr",
            )
        )

    def add_kicker(self) -> None:
        sid = self.counter.alloc()
        self.shapes.append(
            shape_xml(
                sid,
                "kicker",
                720_000,
                260_000,
                2_600_000,
                360_000,
                lines=[self.kicker],
                size=1300,
                color=WHITE,
                fill=ACCENT,
                line=ACCENT,
                bold=True,
                align="ctr",
                anchor="ctr",
            )
        )

    def add_title(self) -> None:
        sid = self.counter.alloc()
        self.shapes.append(
            shape_xml(
                sid,
                "title",
                720_000,
                650_000,
                9_900_000,
                620_000,
                lines=[self.title],
                size=2600,
                color=TEXT,
                bold=True,
            )
        )

    def add_box(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        *,
        lines: Iterable[str],
        fill: str = PANEL,
        line: str = "D9C4A1",
        size: int = 1850,
        color: str = TEXT,
        bold: bool = False,
        align: str = "l",
        anchor: str = "t",
        rounded: bool = True,
        name: str = "box",
    ) -> None:
        sid = self.counter.alloc()
        self.shapes.append(
            shape_xml(
                sid,
                name,
                x,
                y,
                cx,
                cy,
                lines=lines,
                size=size,
                color=color,
                fill=fill,
                line=line,
                bold=bold,
                align=align,
                anchor=anchor,
                rounded=rounded,
            )
        )

    def add_text(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        *,
        lines: Iterable[str],
        size: int = 1850,
        color: str = TEXT,
        bold: bool = False,
        align: str = "l",
        anchor: str = "t",
        name: str = "text",
    ) -> None:
        sid = self.counter.alloc()
        self.shapes.append(
            shape_xml(
                sid,
                name,
                x,
                y,
                cx,
                cy,
                lines=lines,
                size=size,
                color=color,
                bold=bold,
                align=align,
                anchor=anchor,
            )
        )

    def add_arrow(self, x: int, y: int, text: str = "→") -> None:
        self.add_text(
            x,
            y,
            320_000,
            260_000,
            lines=[text],
            size=2600,
            color=ACCENT,
            bold=True,
            align="ctr",
            anchor="ctr",
            name="arrow",
        )

    def build(self) -> str:
        return slide_xml(self.bg, self.shapes)


def slide_xml(bg: str, shapes: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:cSld>"
        "<p:bg><p:bgPr>"
        f'<a:solidFill><a:srgbClr val="{bg}"/></a:solidFill>'
        "<a:effectLst/>"
        "</p:bgPr></p:bg>"
        "<p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        f"{''.join(shapes)}"
        "</p:spTree>"
        "</p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
        "</p:sld>"
    )


def presentation_xml(num_slides: int) -> str:
    master_ref = '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
    slide_ids = "".join(
        f'<p:sldId id="{256 + idx}" r:id="rId{idx + 2}"/>'
        for idx in range(num_slides)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f"{master_ref}<p:sldIdLst>{slide_ids}</p:sldIdLst>"
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )


def presentation_rels_xml(num_slides: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    rels.extend(
        f'<Relationship Id="rId{idx + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx + 1}.xml"/>'
        for idx in range(num_slides)
    )
    rels.append(
        '<Relationship Id="rId50" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>'
    )
    rels.append(
        '<Relationship Id="rId51" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}"
        "</Relationships>"
    )


def content_types_xml(num_slides: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/ppt/slides/slide{idx + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for idx in range(num_slides)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}"
        "</Types>"
    )


ROOT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
    '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
    "</Relationships>"
)


SLIDE_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    "</Relationships>"
)


LAYOUT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
    "</Relationships>"
)


MASTER_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
    "</Relationships>"
)


LAYOUT_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'type="blank" preserve="1">'
    '<p:cSld name="Blank"><p:spTree>'
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
)


MASTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
    '<p:cSld name="Master"><p:spTree>'
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    "</p:spTree></p:cSld>"
    '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
    'accent1="accent1" accent2="accent2" accent3="accent3" '
    'accent4="accent4" accent5="accent5" accent6="accent6" '
    'hlink="hlink" folHlink="folHlink"/>'
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
    "<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>"
    "</p:sldMaster>"
)


THEME_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ParkourTheme">'
    "<a:themeElements>"
    '<a:clrScheme name="ParkourColors">'
    '<a:dk1><a:srgbClr val="1F2937"/></a:dk1>'
    '<a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="111827"/></a:dk2>'
    '<a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>'
    f'<a:accent1><a:srgbClr val="{ACCENT}"/></a:accent1>'
    f'<a:accent2><a:srgbClr val="{ACCENT_2}"/></a:accent2>'
    f'<a:accent3><a:srgbClr val="{ACCENT_3}"/></a:accent3>'
    f'<a:accent4><a:srgbClr val="{ACCENT_4}"/></a:accent4>'
    '<a:accent5><a:srgbClr val="7C3AED"/></a:accent5>'
    '<a:accent6><a:srgbClr val="BE185D"/></a:accent6>'
    '<a:hlink><a:srgbClr val="0563C1"/></a:hlink>'
    '<a:folHlink><a:srgbClr val="954F72"/></a:folHlink>'
    "</a:clrScheme>"
    '<a:fontScheme name="ParkourFonts">'
    '<a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>'
    "</a:fontScheme>"
    '<a:fmtScheme name="ParkourFmt">'
    "<a:fillStyleLst>"
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:gradFill rotWithShape="1"><a:gsLst>'
    '<a:gs pos="0"><a:schemeClr val="phClr"><a:tint val="50000"/><a:satMod val="300000"/></a:schemeClr></a:gs>'
    '<a:gs pos="35000"><a:schemeClr val="phClr"><a:tint val="37000"/><a:satMod val="300000"/></a:schemeClr></a:gs>'
    '<a:gs pos="100000"><a:schemeClr val="phClr"><a:tint val="15000"/><a:satMod val="350000"/></a:schemeClr></a:gs>'
    '</a:gsLst><a:lin ang="16200000" scaled="1"/></a:gradFill>'
    '<a:gradFill rotWithShape="1"><a:gsLst>'
    '<a:gs pos="0"><a:schemeClr val="phClr"><a:shade val="51000"/><a:satMod val="130000"/></a:schemeClr></a:gs>'
    '<a:gs pos="80000"><a:schemeClr val="phClr"><a:shade val="93000"/><a:satMod val="130000"/></a:schemeClr></a:gs>'
    '<a:gs pos="100000"><a:schemeClr val="phClr"><a:shade val="94000"/><a:satMod val="135000"/></a:schemeClr></a:gs>'
    '</a:gsLst><a:lin ang="16200000" scaled="0"/></a:gradFill>'
    "</a:fillStyleLst>"
    "<a:lnStyleLst>"
    '<a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'
    '<a:ln w="25400" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'
    '<a:ln w="38100" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'
    "</a:lnStyleLst>"
    "<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>"
    "<a:bgFillStyleLst>"
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:gradFill rotWithShape="1"><a:gsLst>'
    '<a:gs pos="0"><a:schemeClr val="phClr"><a:tint val="40000"/><a:satMod val="350000"/></a:schemeClr></a:gs>'
    '<a:gs pos="40000"><a:schemeClr val="phClr"><a:tint val="45000"/><a:shade val="99000"/><a:satMod val="350000"/></a:schemeClr></a:gs>'
    '<a:gs pos="100000"><a:schemeClr val="phClr"><a:shade val="20000"/><a:satMod val="255000"/></a:schemeClr></a:gs>'
    '</a:gsLst><a:path path="circle"><a:fillToRect l="50000" t="-80000" r="50000" b="180000"/></a:path></a:gradFill>'
    '<a:gradFill rotWithShape="1"><a:gsLst>'
    '<a:gs pos="0"><a:schemeClr val="phClr"><a:tint val="80000"/><a:satMod val="300000"/></a:schemeClr></a:gs>'
    '<a:gs pos="100000"><a:schemeClr val="phClr"><a:shade val="30000"/><a:satMod val="200000"/></a:schemeClr></a:gs>'
    '</a:gsLst><a:path path="circle"><a:fillToRect l="50000" t="50000" r="50000" b="50000"/></a:path></a:gradFill>'
    "</a:bgFillStyleLst>"
    "</a:fmtScheme>"
    "</a:themeElements>"
    "<a:objectDefaults/><a:extraClrSchemeLst/>"
    "</a:theme>"
)


PRES_PROPS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
    "<p:showPr/></p:presentationPr>"
)


VIEW_PROPS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" lastView="sldView">'
    '<p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr>'
    '<p:slideViewPr><p:cSldViewPr snapToGrid="1" snapToObjects="1" showGuides="1"/></p:slideViewPr>'
    "<p:notesTextViewPr/>"
    '<p:gridSpacing cx="780288" cy="780288"/>'
    "</p:viewPr>"
)


def core_xml(title: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{esc(title)}</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml(slide_titles: list[str]) -> str:
    titles_xml = "".join(f"<vt:lpstr>{esc(title)}</vt:lpstr>" for title in slide_titles)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>OpenAI Codex</Application>"
        "<PresentationFormat>Custom</PresentationFormat>"
        f"<Slides>{len(slide_titles)}</Slides>"
        "<Notes>0</Notes><HiddenSlides>0</HiddenSlides><MMClips>0</MMClips>"
        "<ScaleCrop>false</ScaleCrop>"
        '<HeadingPairs><vt:vector size="2" baseType="variant">'
        "<vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>"
        f"<vt:variant><vt:i4>{len(slide_titles)}</vt:i4></vt:variant>"
        "</vt:vector></HeadingPairs>"
        f'<TitlesOfParts><vt:vector size="{len(slide_titles)}" baseType="lpstr">{titles_xml}</vt:vector></TitlesOfParts>'
        "<Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc>"
        "<HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>"
        "</Properties>"
    )


def build_slides() -> list[tuple[str, str]]:
    slides: list[tuple[str, str]] = []

    s = SlideBuilder("01", "基于当前工作区最新代码路径", "Parkour MoE Estimator-Actor-Critic")
    s.add_text(
        720_000,
        1_420_000,
        10_000_000,
        900_000,
        lines=[
            "核心定位：用 estimator 把 history proprio + proxy depth + vision flag 压成 mcp_code",
            "actor 只看 mcp_code + obs_now，critic 独立看 privileged obs",
            "训练、推理、MuJoCo 和真机部署复用同一套 estimator / actor / selector 结构",
        ],
        size=1850,
        color=MUTED,
    )
    s.add_box(
        720_000,
        2_480_000,
        4_900_000,
        2_050_000,
        lines=[
            "代码主入口",
            "rsl_rl/modules/actor_critic_parkour_moe.py",
            "rsl_rl/algorithms/ppo_parkour_moe.py",
            "rsl_rl/runners/onpolicyrunner_parkour_moe.py",
            "legged_gym/envs/go2/go2_parkour_env.py",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1750,
        color=TEXT,
    )
    s.add_box(
        6_280_000,
        2_480_000,
        5_150_000,
        2_050_000,
        lines=[
            "设计目标",
            "1. 多地形 parkour 下的中间状态建模",
            "2. vision on / off 之间可平滑切换",
            "3. 训练期到部署期的结构一致性",
            "4. 在 easy terrain 上保留 blind fallback 能力",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1750,
    )
    s.add_box(
        720_000,
        5_030_000,
        10_750_000,
        760_000,
        lines=[
            "一句话理解：这版实现本质上是“分层感知控制系统”，不是简单的 PPO + depth concat。",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
        size=1700,
        bold=True,
        anchor="ctr",
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("02", "1. 总体分层", "整套系统不是普通 PPO 加深度拼接", bg=BG_ALT)
    box_w = 1_750_000
    y = 1_720_000
    xs = [720_000, 2_730_000, 4_740_000, 6_750_000, 8_760_000]
    labels = [
        ("Environment", PANEL_SAND, ACCENT_4),
        ("Runner", PANEL, ACCENT),
        ("Estimator", PANEL_TEAL, ACCENT_2),
        ("Actor/Critic", PANEL_BLUE, ACCENT_3),
        ("PPO / Deploy", PANEL_SAND, ACCENT_4),
    ]
    for idx, (label, fill, line) in enumerate(labels):
        s.add_box(xs[idx], y, box_w, 640_000, lines=[label], fill=fill, line=line, size=1700, bold=True, align="ctr", anchor="ctr")
        if idx < len(labels) - 1:
            s.add_arrow(xs[idx] + box_w + 120_000, y + 180_000)
    s.add_box(
        720_000,
        2_830_000,
        5_050_000,
        2_150_000,
        lines=[
            "它解决什么",
            "1. 把高频控制和低频视觉估计解耦",
            "2. 把复杂地形感知压成 actor 可消费的紧凑状态",
            "3. 把视觉失效显式纳入训练闭环，而不是交给数据分布碰运气",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_150_000,
        2_830_000,
        5_320_000,
        2_150_000,
        lines=[
            "设计思路",
            "• 环境负责产生 obs / privileged obs / proxy depth",
            "• runner 负责 history、mask、SwAV window 和训练节奏",
            "• estimator 负责多模态建模与 mcp_code 压缩",
            "• actor/critic 负责决策与价值评估",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("03", "2. 输入与维度", "关键张量都比较固定，结构也很清晰")
    metrics = [
        ("Actor obs", "45"),
        ("Critic obs", "267"),
        ("History", "10 x 45 = 450"),
        ("Depth", "2 x 58 x 87"),
        ("MCP code", "55"),
        ("Action", "12"),
    ]
    card_xs = [720_000, 4_150_000, 7_580_000]
    card_ys = [1_900_000, 3_520_000]
    idx = 0
    for row in card_ys:
        for col in card_xs:
            label, value = metrics[idx]
            s.add_box(
                col,
                row,
                2_930_000,
                1_180_000,
                lines=[label, value],
                fill=PANEL,
                line=ACCENT if idx % 3 == 0 else ACCENT_2 if idx % 3 == 1 else ACCENT_3,
                size=1900 if idx % 2 == 0 else 1750,
                bold=True,
                align="ctr",
                anchor="ctr",
            )
            idx += 1
    s.add_box(
        720_000,
        5_180_000,
        10_780_000,
        800_000,
        lines=[
            "actor_input = mcp_code(55) + obs_now(45) = 100；critic_input = privileged_obs(267)",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1750,
        bold=True,
        anchor="ctr",
        align="ctr",
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("04", "3. 环境与观测来源", "Parkour 环境把可部署观测、特权监督和 proxy depth 分开", bg=BG_ALT)
    s.add_box(
        720_000,
        1_720_000,
        5_050_000,
        1_850_000,
        lines=[
            "Actor 当前观测 45",
            "3 base_ang_vel",
            "3 projected_gravity",
            "3 commands[:3]",
            "12 dof_pos error + 12 dof_vel + 12 previous actions",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_150_000,
        1_720_000,
        5_320_000,
        1_850_000,
        lines=[
            "Critic 特权观测 267",
            "在 45 维基础上追加 base_lin_vel、foot contact、torque、acc、height map",
            "Go2ParkourRobot 又在尾部追加 4 维 feet heights",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        3_900_000,
        5_050_000,
        1_750_000,
        lines=[
            "监督切片",
            "v_t = privileged_obs[0:3]",
            "m_t = privileged_obs[76:263]",
            "h_t = privileged_obs[263:267]",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        6_150_000,
        3_900_000,
        5_320_000,
        1_750_000,
        lines=[
            "运行策略",
            "proxy depth 由高度场 raycast 合成",
            "camera_update_interval = 5 env steps",
            "easy terrain 每 20 次 iteration 翻转一次 vision mask",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("05", "4. Estimator Core", "多模态输入先被 token 化，再交给轻量 Transformer")
    s.add_box(
        720_000,
        1_750_000,
        3_250_000,
        1_350_000,
        lines=[
            "Image branch",
            "2 x 58 x 87",
            "Conv -> Conv -> Conv",
            "64 x 8 x 11 -> 88 tokens",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        4_470_000,
        1_750_000,
        3_250_000,
        1_350_000,
        lines=[
            "Proprio branch",
            "history = 10 x 45",
            "MLP per step -> 64 dim",
            "10 proprio tokens",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        8_220_000,
        1_750_000,
        2_750_000,
        1_350_000,
        lines=[
            "Vision flag",
            "1 -> 64",
            "单独变成一个 token",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        3_520_000,
        10_250_000,
        1_920_000,
        lines=[
            "88 image tokens + 10 proprio tokens + 1 vision token = 99 tokens",
            "再叠加 2D/1D sincos 位置编码和模态类型编码",
            "当 mask_vision = False 时，image tokens 会被同时清零并标成 padding",
            "随后送入 TransformerEncoder(d_model=64, nhead=4, num_layers=1)",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("06", "5. Pooling / Router / Experts", "共享上下文、专家特异关注和共享记忆共同决定状态更新", bg=BG_ALT)
    s.add_box(
        720_000,
        1_720_000,
        4_950_000,
        1_500_000,
        lines=[
            "Shared pooling",
            "1 个 shared_pool_query 从全部有效 tokens 中聚合出 context_feature [B, 64]",
            "它是 router、shared_state update 和所有预测头的公共上下文",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_000_000,
        1_720_000,
        5_470_000,
        1_500_000,
        lines=[
            "Expert pooling",
            "4 个 expert_pool_queries 得到 expert_pooled_features [B, 4, 64]",
            "每个 expert 在共享上下文上再加一层自己的关注偏置",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        3_600_000,
        4_950_000,
        1_900_000,
        lines=[
            "Router",
            "输入 = obs_now(45) + vision_flag(1) + context_feature(64) + shared_state_prev(64)",
            "174 -> 128 -> 4 -> Softmax",
            "输出每个 expert 当前时刻的 gating_weights",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        6_000_000,
        3_600_000,
        5_470_000,
        1_900_000,
        lines=[
            "ReadWriteExpert + shared_state",
            "read(shared_state) -> hidden_init",
            "GRUCell(expert_context, hidden_init) -> hidden_state",
            "write(hidden_state) -> shared_delta",
            "按 gate 加权后再用 shared_state_updater 做统一更新",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("07", "6. Heads 与 MCP", "Estimator 同时做显式状态估计、latent 建模和控制压缩")
    s.add_box(
        720_000,
        1_760_000,
        5_150_000,
        1_900_000,
        lines=[
            "预测头",
            "v_t : 3 维线速度",
            "h_tf : 4 维 feet heights",
            "z_mu / z_logvar : 16 维变分 latent",
            "z_tm : 32 维 terrain latent",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_120_000,
        1_760_000,
        5_350_000,
        1_900_000,
        lines=[
            "解码器",
            "decoder_obs : [v_t, h_tf, z_t] -> o_hat(45)",
            "decoder_map : z_tm -> m_hat(187)",
            "它们分别监督 next proprio 和 terrain map",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        4_070_000,
        10_750_000,
        1_500_000,
        lines=[
            "mcp_code = cat(v_t, h_tf, z_mu, z_tm) = 55",
            "actor 只消费这个紧凑控制表征，而不会直接看 raw depth 或 10 步历史",
            "z_logvar、o_hat、m_hat 参与训练监督，但不直接进入 actor 输入",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1800,
        bold=True,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("08", "7. Actor-Critic", "Actor 看部署输入，Critic 看特权输入，标准不对称训练", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        5_050_000,
        2_050_000,
        lines=[
            "Actor",
            "输入 = mcp_code(55) + obs_now(45) = 100",
            "MLP: 100 -> 512 -> 256 -> 128 -> 12",
            "输出 action mean，并配合可学习 std 形成 Normal 分布",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_150_000,
        1_760_000,
        5_320_000,
        2_050_000,
        lines=[
            "Critic",
            "输入 = privileged_obs(267)",
            "MLP: 267 -> 512 -> 256 -> 128 -> 1",
            "不额外依赖 mcp_code，完全站在训练期特权视角做价值评估",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        4_260_000,
        10_750_000,
        1_350_000,
        lines=[
            "为什么这样分：actor 只学部署时真的拿得到的信息；critic 用 privileged obs 提升 value 估计质量；rollout storage 还会显式保存 mcp_code 供 PPO update 复用。",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
        size=1700,
        bold=True,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("09", "8. 训练数据流", "Estimator 在采样阶段更新，PPO 在 rollout 结束后更新")
    s.add_box(
        720_000,
        1_720_000,
        7_550_000,
        3_760_000,
        lines=[
            "1. reset env，初始化 obs / privileged_obs / history",
            "2. 若 common_step_counter % camera_update_interval == 0，则刷新 estimator",
            "3. estimator 前向得到 est_out，同时 depth autoencoder 做一轮重建更新",
            "4. actor 用当前 mcp_code 采样动作，env.step 后更新 history 与 SwAV window",
            "5. 若本步刷新过 estimator，再用 vt/ht/mt + next obs 做一轮 estimator update",
            "6. 24 个 step 采完后，critic 计算 bootstrap value，再做 PPO actor-critic update",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1700,
    )
    s.add_box(
        8_640_000,
        1_720_000,
        2_830_000,
        1_000_000,
        lines=["num_steps_per_env", "24"],
        fill=PANEL_SAND,
        line=ACCENT_4,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        8_640_000,
        2_980_000,
        2_830_000,
        1_000_000,
        lines=["camera_update_interval", "5"],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        8_640_000,
        4_240_000,
        2_830_000,
        1_240_000,
        lines=["vision_toggle_interval", "20", "仅对 easy terrain 迭代翻转"],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("10", "9. Loss 设计", "三类优化目标分别服务于回报、状态结构和视觉可靠性", bg=BG_ALT)
    s.add_box(
        720_000,
        1_820_000,
        3_250_000,
        2_850_000,
        lines=[
            "PPO 主损失",
            "• surrogate loss",
            "• value loss",
            "• entropy",
            "• adaptive KL 调节 learning rate",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        4_450_000,
        1_820_000,
        3_250_000,
        2_850_000,
        lines=[
            "Estimator 损失",
            "• vt / ht / mt MSE",
            "• next proprio reconstruction",
            "• z KL warmup",
            "• load balance",
            "• terrain SwAV warmup",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        8_180_000,
        1_820_000,
        3_250_000,
        2_850_000,
        lines=[
            "Depth AE",
            "• image reconstruction MSE",
            "• 训练时做自监督视觉约束",
            "• 推理时做 selector / blind fallback 判据",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    s.add_box(
        720_000,
        5_100_000,
        10_710_000,
        620_000,
        lines=["actor-critic / estimator / depth autoencoder 使用三套分离 optimizer，减少目标互相污染。"],
        fill=PANEL_SAND,
        line=ACCENT_4,
        size=1700,
        bold=True,
        anchor="ctr",
        align="ctr",
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("11", "10. 推理与部署", "推理时是低频 estimator + 高频 actor，并带视觉 selector")
    s.add_box(
        720_000,
        1_760_000,
        5_000_000,
        2_100_000,
        lines=[
            "推理 policy",
            "1. done 时重置 estimator hidden state 和 window",
            "2. 更新 proprio history",
            "3. 到刷新周期时再跑 estimator，平时复用 latest_mcp_code",
            "4. actor 直接用 latest_mcp_code + obs 输出动作",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_180_000,
        1_760_000,
        5_280_000,
        2_100_000,
        lines=[
            "selector / blind fallback",
            "easy terrain 上先用 depth autoencoder 估计 recon error",
            "若 recon error > selector_threshold，则 mask_vision = False",
            "于是 estimator 自动切到非视觉 token 路径",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        4_240_000,
        10_740_000,
        1_320_000,
        lines=[
            "导出 bundle 会同时保存 actor_critic / estimator / depth_autoencoder 及其 cfg；MuJoCo 和真机侧适配器都按同一拓扑重建，因此训练部署链路高度一致。",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1700,
        bold=True,
    )
    slides.append((s.title, s.build()))

    s = SlideBuilder("12", "11. 当前代码状态与结论", "这版实现已经形成完整骨架，但还保留一些待激活项", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        5_050_000,
        2_520_000,
        lines=[
            "当前最有价值的设计点",
            "• 感知与控制显式解耦",
            "• shared_state + read/write experts 让切换更平滑",
            "• easy terrain vision toggle + selector 形成 fallback 闭环",
            "• 训练和部署共享一套 estimator / actor / selector",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_150_000,
        1_760_000,
        5_320_000,
        2_520_000,
        lines=[
            "已定义但当前未生效",
            "DefaultEstimator",
            "shared_state_alpha",
            "gate_smooth / cycle_consistency / vision_consistency",
            "gt_mt_step、mcp_code_additional",
            "fc_zt / map_unet / fc_zm_fine",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        720_000,
        4_700_000,
        10_750_000,
        860_000,
        lines=[
            "最终结论：最新代码的本质是“层级化感知控制系统”，而不是简单的 PPO + depth concat。",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1750,
        bold=True,
        anchor="ctr",
        align="ctr",
    )
    slides.append((s.title, s.build()))

    return slides


def write_pptx(out_path: Path) -> None:
    slides = build_slides()
    slide_titles = [title for title, _ in slides]

    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(slides)))
        zf.writestr("_rels/.rels", ROOT_RELS_XML)
        zf.writestr("docProps/core.xml", core_xml(slide_titles[0]))
        zf.writestr("docProps/app.xml", app_xml(slide_titles))
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml(len(slides)))
        zf.writestr("ppt/presProps.xml", PRES_PROPS_XML)
        zf.writestr("ppt/viewProps.xml", VIEW_PROPS_XML)
        zf.writestr("ppt/theme/theme1.xml", THEME_XML)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", MASTER_XML)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS_XML)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT_XML)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS_XML)
        for idx, (_, slide_xml_str) in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml_str)
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", SLIDE_RELS_XML)


if __name__ == "__main__":
    output = Path(__file__).with_name("parkour_moe_estimator_actor_critic_framework_zh.pptx")
    write_pptx(output)
    print(f"Generated: {output}")
