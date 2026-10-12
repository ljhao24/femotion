"""Chinese wording of the JTG checks for calculation reports (femotion.codes.jtg.report): check names, the program's
messages (notes, warnings, conditions not met), the rules of the detailing reports and the names of the values in
Check.details. Patterns are regular expressions on the program's English text; numbers and clause numbers are kept as
they are. Names: the first pattern that matches wins (specific before general). A text no pattern covers is returned
unchanged (tests/test_report.py lists every text of the library's checks and requires a translation)."""
from __future__ import annotations

import re

# words that appear inside names and messages
WORDS = {
    "top": "上缘", "bottom": "下缘", "precast top": "预制梁顶缘", "precast bottom": "预制梁底缘", "topping top": "现浇层顶缘",
    "topping bottom": "现浇层底缘", "tension": "受拉", "compression": "受压", "full prestress": "全预应力构件", "class A": "A 类构件",
    "class B": "B 类构件", "precast girder": "预制梁", "composite member": "组合构件", "superstructure": "上部结构",
    "substructure": "下部结构", "foundation": "基础", "flexural": "受弯", "small eccentricity": "小偏心",
    "large eccentricity": "大偏心", "axial tension": "轴心受拉", "eccentric tension": "偏心受拉",
    "eccentric tension (circle)": "偏心受拉（圆形截面）", "positive": "正温差", "negative": "反温差", "none": "不计温差", "sagging": "正弯矩",
    "hogging": "负弯矩", "tension face": "受拉面", "far face": "远离轴力一侧", "I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ", "V": "Ⅴ",
    "VI": "Ⅵ"
}


def word(s: str) -> str:
    return WORDS.get(s, s)


def clause(s: str) -> str:
    """'6.3.1 item 1-3' → '6.3.1 第 1 款第 3 项', 'table 9.1.1' → '表 9.1.1', ', should' → '，宜' …"""
    s = re.sub(r"\btable (\S+)", r"表 \1", s)
    s = re.sub(r"\bappendix (\S+)", r"附录 \1", s)
    s = re.sub(r"(\S+) items? (\d+)-(\d+)", r"\1 第 \2 款第 \3 项", s)
    s = re.sub(r"(\S+) items (\d+), (\d+)", r"\1 第 \2、\3 款", s)
    s = re.sub(r"(\S+) item (\d+)", r"\1 第 \2 款", s)
    s = re.sub(r"\bnote (\d+)", r"注 \1", s)
    s = s.replace("commentary to ", "条文说明 ").replace("commentary", "条文说明")
    s = s.replace(", should", "，宜").replace("; should", "，宜").replace("'should'", "宜").replace(", may", "，可")
    s = s.replace("program convention", "程序约定")
    return s.replace(", ", "、").replace("; ", "；")


def _g(mt, k):
    v = mt.group(k)
    return word(v) if v is not None else ""


# ── names ───────────────────────────────────────────────────────────────────
# (pattern, zh); {k} is group k through WORDS; the first match wins
NAMES = [
    (r"flexure \(hogging\)", "正截面抗弯承载力（负弯矩）"),
    (r"flexure", "正截面抗弯承载力"),
    (r"shear \(slab without stirrups\)", "不配箍筋板的斜截面抗剪"),
    (r"shear \(circular section\)", "圆形截面斜截面抗剪"),
    (r"shear \(composite member\)", "组合构件斜截面抗剪"),
    (r"shear \(precast girder, stage (\d+)\)", "预制梁斜截面抗剪（第 {1} 阶段）"),
    (r"shear \(hogging side\)", "斜截面抗剪承载力（负弯矩区）"),
    (r"shear share ξ = (\S+) of the concrete and stirrups", "箍筋设计回代（混凝土和箍筋承担 ξ = {1}）"),
    (r"shear", "斜截面抗剪承载力"),
    (r"section limit at the end support x = (\S+)", "端支点截面尺寸限制（x = {1} m）"),
    (r"axial compression \(spiral column\)", "螺旋箍筋柱轴心受压承载力"),
    (r"axial compression", "轴心受压承载力"),
    (r"biaxial eccentric compression \(circle\)", "双向偏心受压承载力（圆形截面）"),
    (r"biaxial eccentric compression", "双向偏心受压承载力"),
    (r"biaxial eccentric tension", "双向偏心受拉承载力"),
    (r"eccentric compression", "偏心受压承载力"),
    (r"tension \(hogging side\)", "受拉承载力（负弯矩侧）"),
    (r"tension", "受拉承载力"),
    (r"torsion with shear \(web\)", "剪扭承载力（腹板）"),
    (r"torsion with shear", "剪扭承载力"),
    (r"torsion \(web\)", "受扭承载力（腹板）"),
    (r"torsion \((top|bottom) flange\)", "受扭承载力（{1}翼缘）"),
    (r"torsion", "受扭承载力"),
    (r"crack width \(composite\)", "裂缝宽度（组合构件）"),
    (r"crack width", "裂缝宽度"),
    (r"deflection \(long term, live load\)", "长期挠度（汽车荷载）"),
    (r"crack resistance \(normal section, full prestress, (top|bottom) edge\)", "正截面抗裂（全预应力构件，{1}）"),
    (r"crack resistance \(normal section, class A, (top|bottom) edge\)", "正截面抗裂（A 类构件，{1}）"),
    (r"crack resistance \(normal section, class B under self-weight, (top|bottom) edge\)", "正截面抗裂（B 类构件，自重作用，{1}）"),
    (r"crack resistance \(inclined section, composite\)", "斜截面抗裂（组合构件）"),
    (r"crack resistance \(inclined section\)", "斜截面抗裂"),
    (r"crack resistance \((precast girder|composite member), (full prestress|class A), (precast top|precast "
     r"bottom|topping top|topping bottom)\)",
     "正截面抗裂（{1}，{2}，{3}）"),
    (r"crack resistance \((precast girder|composite member), class B under self-weight, (precast top|precast "
     r"bottom|topping top|topping bottom)\)",
     "正截面抗裂（{1}，B 类构件自重作用，{2}）"),
    (r"principal compressive stress \(service, composite\)", "使用阶段混凝土主压应力（组合构件）"),
    (r"principal compressive stress \(service\)", "使用阶段混凝土主压应力"),
    (r"concrete compression in service \(cracked section\)", "使用阶段混凝土法向压应力（开裂截面）"),
    (r"concrete compression in service \(composite, (precast top|precast bottom|topping top|topping bottom)\)",
     "使用阶段混凝土法向压应力（组合构件，{1}）"),
    (r"concrete compression in service \((top|bottom) edge\)", "使用阶段混凝土法向压应力（{1}）"),
    (r"transient stresses \(reinforced concrete\)", "短暂状况应力（钢筋混凝土）"),
    (r"transient shear stresses \(reinforced concrete\)", "短暂状况剪应力（钢筋混凝土）"),
    (r"transient stresses \(prestressed, (top|bottom) edge in compression\)", "短暂状况应力（预应力，{1}受压）"),
    (r"local compression \(no indirect reinforcement\)", "局部承压（未配间接钢筋）"),
    (r"local compression", "局部承压"),
    (r"punching \(slab, 5\.6\.1\)", "冲切（板，不配抗冲切钢筋）"),
    (r"punching \(slab with punching steel, 5\.6\.2\)", "冲切（板，配抗冲切钢筋）"),
    (r"punching \(footing, 5\.6\.3\)", "冲切（扩大基础）"),
    (r"pile cap punching \(column\)", "承台冲切（墩柱）"),
    (r"pile cap punching \(corner pile\)", "承台冲切（角桩）"),
    (r"pile cap punching \(edge pile\)", "承台冲切（边桩）"),
    (r"pile cap shear", "承台斜截面抗剪"),
    (r"pile cap strut (\S+)", "承台撑杆 {1}"),
    (r"pile cap tie", "承台拉杆"),
    (r"pile cap detailing \((.*)\)", "承台构造"),
    (r"cap beam flexure \(hogging\)", "盖梁正截面抗弯（负弯矩）"),
    (r"cap beam flexure", "盖梁正截面抗弯"),
    (r"cap beam shear", "盖梁斜截面抗剪"),
    (r"cap beam detailing \((.*)\)", "盖梁构造"),
    (r"cap cantilever top tie", "盖梁悬臂顶部拉杆"),
    (r"top tie", "顶部拉杆"),
    (r"diagonal strut", "斜撑杆"),
    (r"node (\d+) bearing face", "节点 {1} 支承面"),
    (r"node (\d+) face of strut (\d+) \((\S+)\)", "节点 {1} 与撑杆 {2}（节点 {3}）的接触面"),
    (r"strut (\d+) \((\S+)\)", "撑杆 {1}（节点 {2}）"),
    (r"tie (\d+) \((\S+)\)", "拉杆 {1}（节点 {2}）"),
    (r"surface mesh \(B\.3\.5\)", "表层钢筋网（B.3.5）"),
    (r"bursting steel \((\d+) legs φ(\S+)\)", "劈裂力钢筋（{1} 肢 φ{2}）"),
    (r"bursting steel in the web", "腹板劈裂力钢筋"),
    (r"composite joint shear \(beam\)", "叠合面抗剪（梁）"),
    (r"composite joint shear \(slab\)", "叠合面抗剪（板）"),
    (r"bearing area \(8\.7\.3-1\)", "支座有效承压面积"),
    (r"rubber thickness, shear without braking \((\S+)\)", "橡胶层总厚度（剪切变形，不计制动力）"),
    (r"rubber thickness, shear with braking \((\S+)\)", "橡胶层总厚度（剪切变形，计入制动力）"),
    (r"rubber thickness, stability \((\S+)\)", "橡胶层总厚度（稳定）"),
    (r"compression of the bearing \(8\.7\.3-8, -9\)", "支座竖向平均压缩变形"),
    (r"steel plates \(8\.7\.3-10\)", "加劲钢板厚度"),
    (r"sliding without braking \(8\.7\.4-1\)", "支座抗滑稳定（不计制动力）"),
    (r"sliding with braking \(8\.7\.4-2\)", "支座抗滑稳定（计入制动力）"),
    (r"PTFE friction without braking \(8\.7\.5-1\)", "四氟滑板支座摩阻力（不计制动力）"),
    (r"PTFE friction with braking \(8\.7\.5-2\)", "四氟滑板支座摩阻力（计入制动力）"),
    (r"concrete hinge, compression \(8\.6\.1\)", "混凝土铰抗压"),
    (r"concrete hinge, transverse tension \(8\.6\.2\)", "混凝土铰横向受拉"),
    (r"expansion joint \(8\.8\.2\)", "伸缩装置伸缩量"),
    (r"lifting loops \(9\.8\.2\)", "吊环应力"),
    (r"in-plane principal tension \(table 6\.1\.3\)", "箱梁面内主拉应力"),
    (r"in-plane principal compression", "箱梁面内主压应力"),
    (r"transverse normal stress \((full prestress|class A), (top|bottom) face\)", "横向正应力（{1}，{2}）"),
    (r"durability, environment (\S+) \(4\.5\)", "耐久性（{1} 类环境）"),
    (r"section detailing (.*)", "截面构造"),
    (r"stirrups at a curved soffit \(9\.3\.14\)", "曲线形底面处箍筋"),
    (r"stirrups \(9\.3\.12, 9\.1\.6\)", "箍筋构造"),
    (r"bars over an end support \(9\.3\.9\)", "端支点处纵筋构造"),
    (r"lap splice of (\S+) mm (\S+) \((.*)\)", "{2} Φ{1} 钢筋绑扎搭接"),
    (r"welded splice of (\S+) mm (\S+) \((.*)\)", "{2} Φ{1} 钢筋焊接接头"),
    (r"mechanical splice of (\S+) mm (\S+) \((.*)\)", "{2} Φ{1} 钢筋机械连接接头"),
    (r"cut-off of a (\S+) mm bar \(9\.3\.8\)", "Φ{1} 钢筋截断"),
    (r"bent-up bar of (\S+) mm \(9\.3\.10\)", "Φ{1} 弯起钢筋"),
    (r"welded frame \(9\.3\.11\)", "焊接骨架"),
    (r"girder top slab \((.*)\)", "梁顶板（翼缘板）构造"),
    (r"box bottom slab \(9\.3\.5\)", "箱梁底板构造"),
    (r"girder dimensions \(9\.3\.2\)", "梁的尺寸"),
    (r"box detailing \(9\.3\.2\)", "箱梁构造"),
    (r"diaphragms of a (\S+) superstructure \(9\.3\.1\)", "{1} 梁上部结构的横隔梁"),
    (r"composite beam \((.*)\)", "组合梁构造"),
    (r"two-way slab \(9\.2\.5\)", "双向板构造"),
    (r"skew slab, (\S+)° \(9\.2\.6\)", "斜板构造（斜交角 {1}°）"),
    (r"composite slab \(9\.2\.7\)", "组合板构造"),
    (r"hinged slabs \(9\.2\.8\)", "铰接板构造"),
    (r"slab detailing \(9\.2\)", "板的构造"),
    (r"bearing detailing \(9\.7\)", "支座构造"),
    (r"culvert detailing \(9\.8\.1\)", "涵洞构造"),
    (r"column detailing \((.*)\)", "墩柱构造"),
    (r"column bars in a joint \((.*)\)", "节点内墩柱纵筋"),
    (r"beam bars in a joint \((.*)\)", "节点内梁纵筋"),
    (r"prestressed girder detailing \((.*)\)", "预应力梁构造"),
    (r"duct detailing \((.*)\)", "管道构造"),
    (r"curved duct cover \((.*)\)", "曲线管道保护层"),
    (r"curved duct spacing \((.*)\)", "曲线管道间距"),
    (r"end anchorage zone detailing \((.*)\)", "端部锚固区构造"),
    (r"external tendon detailing", "体外束构造"),
    (r"a rule", "构造规则"),
]


def name_zh(name: str) -> str | None:
    for pat, zh in NAMES:
        mt = re.fullmatch(pat, name)
        if mt:
            return re.sub(r"\{(\d+)\}", lambda k: word(mt.group(int(k.group(1)))), zh)
    return None


# ── rules of the detailing reports ──────────────────────────────────────────
FACE = {
    "bottom": "底", "top": "顶"
}
RULES = [
    (r"cover of the (bottom|top) bars", "{f}排钢筋保护层"),
    (r"cover of the outermost steel at the (bottom|top)", "{f}面最外侧钢筋保护层"),
    (r"cover of the outermost steel at the sides", "侧面最外侧钢筋保护层"),
    (r"side cover of the (bottom|top) bars", "{f}排钢筋侧面保护层"),
    (r"clear spacing of the (bottom|top) bars at depth (\S+)", "{f}排钢筋净距（深 {2} mm）"),
    (r"clear distance of the (bottom|top) layers at depths (\S+)", "{f}排钢筋层间净距（深 {2} mm）"),
    (r"layers of the welded frame \((bottom|top)\)", "{f}排焊接骨架层数"),
    (r"bars of the welded frame \((bottom|top)\)", "{f}排焊接骨架钢筋直径"),
    (r"side bars of the webs, As/\(b h\)", "腹板侧面钢筋配筋率 As/(bh)"),
    (r"spacing of the side bars, (tension|compression) zone", "腹板侧面钢筋间距（{z}区）"),
    (r"cover of the ducts \(top, bottom\)", "管道保护层（顶、底）"),
    (r"side cover of the ducts", "管道侧面保护层"),
    (r"cover of the pretensioned tendons \(top, bottom\)", "先张预应力筋保护层（顶、底）"),
    (r"side cover of the pretensioned tendons", "先张预应力筋侧面保护层"),
    (r"cover of the longitudinal bars", "纵筋保护层"),
    (r"cover of the outermost steel", "最外侧钢筋保护层"),
    (r"layers of the frame", "骨架层数"),
    (r"bar diameter of the frame", "骨架钢筋直径"),
    (r"weld of the inclined bars", "斜筋焊缝长度"),
    (r"short weld between longitudinal bars", "纵筋间短焊缝长度"),
    (r"bar diameter of a lap splice", "绑扎搭接钢筋直径"),
    (r"lap length", "搭接长度"),
    (r"distance between laps of one bar", "同一钢筋相邻搭接接头间距"),
    (r"clear distance at the lap", "搭接处钢筋净距"),
    (r"stirrup diameter \(≥ main bar / 4\)", "箍筋直径（不小于主筋直径的 1/4）"),
    (r"stirrup diameter", "箍筋直径"),
    (r"stirrup ratio ρsv", "配箍率 ρsv"),
    (r"stirrup spacing along a tension lap", "受拉搭接范围内箍筋间距"),
    (r"stirrup spacing within h of the support", "支座中心一倍梁高范围内箍筋间距"),
    (r"stirrup spacing", "箍筋间距"),
    (r"stirrup hook bend diameter", "箍筋弯钩弯弧内直径"),
    (r"stirrup hook straight part", "箍筋弯钩平直段长度"),
    (r"stirrup leg area Asv1", "单肢箍筋面积 Asv1"),
    (r"bottom bars through the support", "伸过支座的底部纵筋根数"),
    (r"extension past the support", "伸过支座的长度"),
    (r"deviation radius", "转向半径"),
    (r"free length", "体外束自由长度"),
    (r"inner hoop diameter", "内层箍筋直径"),
    (r"hoop spacing", "箍筋间距"),
    (r"anchorage crossbeam thickness", "锚固横梁厚度"),
    (r"web stirrup diameter", "腹板箍筋直径"),
    (r"web stirrup spacing", "腹板箍筋间距"),
    (r"length of the dense end zone from the support centre", "支座中心起箍筋加密段长度"),
    (r"bulb stirrup diameter", "马蹄箍筋直径"),
    (r"bulb stirrup spacing", "马蹄箍筋间距"),
    (r"ribbed web stirrups", "腹板带肋箍筋"),
    (r"closed stirrups near the support", "支座附近封闭箍筋"),
    (r"closed bulb stirrups", "马蹄封闭箍筋"),
    (r"closed bursting stirrups", "封闭劈裂力箍筋"),
    (r"clear spacing of straight ducts", "直线管道净距"),
    (r"clear spacing / duct diameter", "管道净距 / 管道直径"),
    (r"ducts stacked vertically", "竖向叠放的管道数"),
    (r"inner duct area / tendon area", "管道内截面面积 / 预应力钢筋面积"),
    (r"tendon row (\d+) \(depth (\S+) mm\): inner duct area / tendon area", "第 {1} 排钢束（深 {2} mm）管道面积 / 钢束面积"),
    (r"tendon radius", "钢束曲线半径"),
    (r"tendon radius \(give radius, d and the tendon kind\)", "钢束曲线半径（需给出半径、管道直径和钢束种类）"),
    (r"(top|bottom) slab, cell (\d+): ln/30", "{f}板 ln/30（第 {2} 室，自 0 起计）"),
    (r"straight length behind the anchor", "锚具后直线段长度"),
    (r"grout strength", "压浆强度"),
    (r"cover out of the plane of the curve", "曲线平面外保护层"),
    (r"cover in the plane of the curve", "曲线平面内保护层"),
    (r"clear spacing in the plane of the curve", "曲线平面内管道净距"),
    (r"clear spacing out of the plane", "曲线平面外管道净距"),
    (r"clear spacing in the plane / duct diameter", "曲线平面内管道净距 / 管道直径"),
    (r"clear spacing out of the plane / duct diameter", "曲线平面外管道净距 / 管道直径"),
    (r"ρv under the plate", "锚垫板下间接钢筋体积配筋率 ρv"),
    (r"spiral length under the plate \(≥ trumpet length\)", "锚垫板下螺旋筋长度（不小于喇叭管长度）"),
    (r"bursting stirrup spacing", "劈裂力箍筋间距"),
    (r"sealing concrete class", "封锚混凝土强度等级"),
    (r"bars against spalling at the end face", "端面防剥裂钢筋"),
    (r"bars round the sealed anchors", "封锚处钢筋"),
    (r"trumpet anchor plate", "喇叭管锚垫板"),
    (r"longitudinal bar diameter", "纵筋直径"),
    (r"least clear spacing of longitudinal bars", "纵筋最小净距"),
    (r"largest clear spacing of longitudinal bars", "纵筋最大净距"),
    (r"longitudinal steel ratio", "纵筋配筋率"),
    (r"embedment into the foundation / cap beam \(≥ la\)", "纵筋伸入基础或盖梁的长度（不小于 la）"),
    (r"tie diameter", "箍筋直径"),
    (r"tie spacing", "箍筋间距"),
    (r"closed ties", "封闭箍筋"),
    (r"ties \(give ties = \(diameter, spacing\)\)", "箍筋（需给出直径和间距）"),
    (r"anchorage from the cap soffit \(≥ la\)", "自盖梁底面起的锚固长度（不小于 la）"),
    (r"concrete class", "混凝土强度等级"),
    (r"side bar diameter", "侧面钢筋直径"),
    (r"side bar spacing", "侧面钢筋间距"),
    (r"cap depth \(≥ 1\.5 m\)", "承台厚度（不小于 1.5 m）"),
    (r"cap depth \(≥ 1\.5 D\)", "承台厚度（不小于 1.5D）"),
    (r"mesh over the pile heads, direction (\d+)", "桩顶钢筋网（第 {1} 方向）"),
    (r"mesh bar diameter \(lower end\)", "钢筋网直径（下限）"),
    (r"mesh bar diameter \(upper end\)", "钢筋网直径（上限）"),
    (r"vertical tie diameter", "竖向箍筋直径"),
    (r"bars up to the cap top", "伸至承台顶面的钢筋"),
    (r"top bars through the joint", "穿过节点的顶部钢筋"),
    (r"splices outside the joint", "节点外的钢筋接头"),
    (r"mixed reinforcement", "混合配筋"),
    (r"weld length", "焊缝长度"),
    (r"area of the splice bars", "帮条钢筋面积"),
    (r"distance between joints of one bar", "同一钢筋相邻接头间距"),
    (r"share of welded bars in the zone", "区段内焊接接头钢筋比例"),
    (r"clear distance at the welded joint", "焊接接头处钢筋净距"),
    (r"cover of the couplers against table 9\.1\.1", "连接件保护层（对照表 9.1.1）"),
    (r"cover of the couplers", "连接件保护层"),
    (r"bend start beyond the fully used section", "弯起点至充分利用截面的距离"),
    (r"end anchorage in the tension zone", "受拉区末端锚固长度"),
    (r"(longitudinal|transverse) bars / slab section", "{d}钢筋面积 / 板截面面积"),
    (r"(longitudinal|transverse) layer (\d+) diameter", "{d}第 {2} 层钢筋直径"),
    (r"(longitudinal|transverse) layer (\d+) spacing", "{d}第 {2} 层钢筋间距"),
    (r"flange at the web", "腹板处翼缘厚度"),
    (r"diaphragm spacing", "横隔梁间距"),
    (r"cast-in-place layer at the joint", "接缝处现浇层厚度"),
    (r"roughness of the precast top", "预制件顶面粗糙度"),
    (r"stirrups into the deck", "箍筋伸入桥面板长度"),
    (r"edge strip bars / middle strip bars", "边带钢筋 / 中带钢筋"),
    (r"middle strip spacing", "中带钢筋间距"),
    (r"edge strip spacing", "边带钢筋间距"),
    (r"connector embedment in the precast slab", "连接钢筋伸入预制板长度"),
    (r"connector embedment in the topping", "连接钢筋伸入现浇层长度"),
    (r"connector spacing", "连接钢筋间距"),
    (r"hinge groove depth", "铰缝深度"),
    (r"topping over hinged slabs", "铰接板上现浇层厚度"),
    (r"embedment into the foundation / cap beam \(≥ la\)", "纵筋伸入基础或盖梁的长度（不小于 la）"),
    (r"column bars through tie-beam joints", "墩柱纵筋穿过系梁节点"),
    (r"cover in the plane of the curve", "曲线平面内保护层"),
]


def rule_zh(name: str) -> str | None:
    for pat, zh in RULES:
        mt = re.fullmatch(pat, name)
        if mt:
            g1 = mt.group(1) if mt.groups() else ""
            out = zh.replace("{f}", FACE.get(g1, "")).replace("{z}", {"tension": "受拉", "compression": "受压"}.get(g1, ""))
            out = out.replace("{d}", {"longitudinal": "纵向", "transverse": "横向"}.get(g1, ""))
            return re.sub(r"\{(\d+)\}", lambda k: mt.group(int(k.group(1))), out)
    return None


# ── messages ────────────────────────────────────────────────────────────────
N = r"(−?-?[\d.]+)"
MESSAGES = [
    (r"C(\d+) with (\S+): at least C30 for bars of 400 MPa and above \(3\.1\.2\)",
     "C{1} 混凝土配 {2} 钢筋：采用 400 MPa 及以上钢筋时混凝土不应低于 C30（3.1.2）"),
    (r"x = (\S+) < 2a' = (\S+) mm: compression bars ignored \(singly reinforced\) governs \((\S+)\)",
     "x = {1} mm < 2a′ = {2} mm：不计受压钢筋（单筋截面）的承载力较大，取用（{3}）"),
    (r"x = (\S+) < 2a' = (\S+) mm: moments about the compression steel governs \((\S+)\)",
     "x = {1} mm < 2a′ = {2} mm：对受压钢筋合力点取矩的承载力较大，取用（{3}）"),
    (r"x = (\S+) < 2a' = (\S+) mm: compression bars ignored governs \(5\.4\.2 → 5\.2\.4, e' = (\S+) mm\)",
     "x = {1} mm < 2a′ = {2} mm：不计受压钢筋的承载力较大，取用（5.4.2 → 5.2.4，e′ = {3} mm）"),
    (r"x = (\S+) < 2a' = (\S+) mm: moments about the compression steel governs \(5\.4\.2 → 5\.2\.4, e' = (\S+) mm\)",
     "x = {1} mm < 2a′ = {2} mm：对受压钢筋合力点取矩的承载力较大，取用（5.4.2 → 5.2.4，e′ = {3} mm）"),
    (r"x = (\S+) < 2a's = (\S+) mm: moments about A's, z = k \(h0 − a's\) \((.*)\)",
     "x = {1} mm < 2a′s = {2} mm：对 A′s 取矩，z = k(h0 − a′s)（{3}）"),
    (r"x < 2a': moments about the compression steel \(e' = (\S+) mm, 5\.3\.6\)",
     "x < 2a′：对受压钢筋合力点取矩（e′ = {1} mm，5.3.6）"),
    (r"the shear at the support centre \(V0d\), b and h0 of the support section \(program convention, the "
     r"near-end-support segment of 5\.2\.11\)",
     "取支点中心处剪力 V0d 和支点截面的 b、h0（程序约定：5.2.11 的端支点附近梁段）"),
    (r"γ0Vd ≤ (\S+) kN: detailing stirrups per 9\.3\.12 suffice \(5\.2\.12\)",
     "γ0Vd ≤ {1} kN：可只按 9.3.12 构造配置箍筋（5.2.12）"),
    (r"γ0Vd ≤ (\S+) kN: detailing hoops suffice \(5\.2\.12\)", "γ0Vd ≤ {1} kN：可只按构造配置箍筋（5.2.12）"),
    (r"α2 = (\S+) given", "α2 = {1}（给定）"),
    (r"α1 = (\S+) \(near an end support, 5\.2\.9\)", "α1 = {1}（端支点附近，5.2.9）"),
    (r"α1 = (\S+) \(near an interior support, 5\.2\.9\)", "α1 = {1}（中间支点附近，5.2.9）"),
    (r"class B \(cracking allowed\): α2 = 1\.0", "B 类构件（允许开裂）：α2 = 1.0"),
    (r"prestressing moment in the direction of the external moment: α2 = 1\.0", "预加力弯矩与外弯矩同向：α2 = 1.0"),
    (r"Vcs of the composite \((\S+) kN\) below the precast girder's \((\S+) kN\): the latter taken \(8\.1\.5\)",
     "组合截面的 Vcs（{1} kN）小于预制梁的（{2} kN），取后者（8.1.5）"),
    (r"8\.1\.6 holds with the detailing of 9\.3\.16, 9\.3\.17: check it with "
     r"jtg\.detailing\.composite_beam_detailing \(warnings\)",
     "8.1.6 以满足 9.3.16、9.3.17 的构造为前提，构造另行检查（组合梁构造）"),
    (r"σtp = (\S+) ≤ 0\.5 ftk = (\S+) MPa: detailing stirrups \(7\.1\.6\)",
     "σtp = {1} ≤ 0.5ftk = {2} MPa：箍筋可按构造设置（7.1.6）"),
    (r"σtp = (\S+) > 0\.5 ftk: stirrup spacing by 7\.1\.6-2 \(pass stirrups to check\)",
     "σtp = {1} > 0.5ftk：箍筋间距按式（7.1.6-2）确定（未给出箍筋，未检查）"),
    (r"stirrup spacing (\S+) mm > fsk Asv / \(σtp b\) = (\S+) mm \(7\.1\.6-2\)",
     "箍筋间距 {1} mm > fskAsv/(σtp b) = {2} mm（7.1.6-2）"),
    (r"(top|bottom|precast top|precast bottom|topping top|topping bottom): σlt = (\S+) > σpc = (\S+) MPa "
     r"\(6\.3\.1-4\)",
     "{1}：σlt = {2} > σpc = {3} MPa（6.3.1-4）"),
    (r"(top|bottom|precast top|precast bottom|topping top|topping bottom): (\S+) = (\S+) > (\S+) MPa",
     "{1}：{2} = {3} > {4} MPa"),
    (r"4\.5\.4: cover per 9\.1\.1 \(section_detailing\); impermeability and frost-resistance grades where required; "
     r"forms and details that drain and ventilate \(not checked by numbers\)",
     "4.5.4：保护层按 9.1.1 检查（截面构造）；需要时的抗渗、抗冻等级以及利于排水、通风的构造不作数值检查"),
    (r"9\.3\.3 horizontal clear spacing: bars spread evenly between the stirrups across the width at their depth, "
     r"the largest the layout allows \(program convention\)",
     "9.3.3 水平净距：同排钢筋在箍筋内沿该深度的宽度均匀布置，取布置所能达到的最大净距（程序约定）"),
    (r"9\.3\.7 tension zone (below|above) depth (\S+), the cracked section's axis \(reinforced concrete\); a gap "
     r"reaching into it takes the tension limit",
     "9.3.7 受拉区取开裂截面中和轴（距顶面 {2} mm）以{b}；跨入受拉区的间距按受拉区限值检查"),
    (r"9\.3\.7 tension zone (below|above) depth (\S+), the gross centroid \(prestressed\); a gap reaching into it "
     r"takes the tension limit",
     "9.3.7 受拉区取毛截面重心（距顶面 {2} mm）以{b}；跨入受拉区的间距按受拉区限值检查"),
    (r"side faces not checked \(side_cover=…\); the horizontal room takes the bottom cover of the outermost steel",
     "未给出侧面保护层，侧面不检查；水平方向按底面最外侧钢筋的保护层布置"),
    (r"the two outer bars run past the support, turn up at a right angle and join the top bars \(9\.3\.9\)",
     "两侧外边的钢筋伸过支座后直角弯起并与顶部钢筋相连（9.3.9）"),
    (r"tu = t0 \+ 3650 d \(the 10-year final values of commentary tables C-1, C-2\)",
     "tu = t0 + 3650 d（条文说明表 C-1、C-2 的 10 年终值）"),
    (r"external tendons at σpe,ex \(5\.2\.5\)", "体外束按 σpe,ex 计（5.2.5）"),
    (r"block deviators need both inner closed hoops \(round each deviator\) and outer ones \(round all of them, "
     r"9\.4\.25\)",
     "块式转向构造须设内层封闭箍筋（每个转向孔各一道）和外层箍筋（包住全部转向孔，9.4.25）"),
    (r"σss = (\S+) MPa ≤ 0 \(z = (\S+) mm\): no crack opening under Ms", "σss = {1} MPa ≤ 0（z = {2} mm）：Ms 下裂缝不张开"),
    (r"σst = (\S+) ≤ σpc = (\S+) MPa: the tension face is not decompressed, no crack",
     "σst = {1} ≤ σpc = {2} MPa：受拉边缘未消压，不开裂"),
    (r"e0/h = (\S+) ≤ 0\.55: crack width need not be checked \(6\.4\.3\)", "e0/h = {1} ≤ 0.55：可不验算裂缝宽度（6.4.3）"),
    (r"e0/r = (\S+) ≤ 0\.55: crack width need not be checked \(6\.4\.3\)", "e0/r = {1} ≤ 0.55：可不验算裂缝宽度（6.4.3）"),
    (r"bars in the pretensioned zone up to φ(\S+): at most 14 mm recommended \(7\.2\.8, 'should'\)",
     "预拉区钢筋直径达 φ{1}：宜不大于 14 mm（7.2.8，宜）"),
    (r"9\.4\.2: the tension-face bars preferably ribbed, of small diameter, closely spaced \(not checked\)",
     "9.4.2：受拉面钢筋宜用带肋、小直径、较密的钢筋（未检查）"),
    (r"9\.4\.10 commentary: special ducts \(e\.g\. semicircular hoop tendons of cable-stayed pylons, radius about "
     r"1\.5 m\) may go below with special measures",
     "9.4.10 条文说明：特殊管道（如斜拉桥索塔的半圆形环向钢束，半径约 1.5 m）采取专门措施后可以更小"),
    (r"9\.4\.9 item 4: ducts take the pre-camber where the member has one \(not checked\)",
     "9.4.9 第 4 款：构件有预拱度时管道随之设置（未检查）"),
    (r"9\.4\.9 item 3: the section's duct diameter taken as the inner diameter; it is normally the outer one "
     r"\(deducted for the net section\), which overstates the inner area: pass d_inner",
     "9.4.9 第 3 款：截面数据中的管道直径按内径计；通常给的是外径（净截面扣除用），会高估管道内截面面积，应另给内径"),
    (r"clear spacing / duct diameter (\S+) < 0\.6 \(9\.4\.9 item 1; should\)", "管道净距 / 管道直径 {1} < 0.6（9.4.9 第 1 款，宜）"),
    (r"skipped \(no input\): (.*)", None),  # handled in message()
    (r"\(A's \+ A'p\)/A = (\S+) < (\S+) in the pretensioned zone \(7\.2\.8; required by the code, minimum "
     r"reinforcement reported as a warning\)",
     "预拉区 (A′s + A′p)/A = {1} < {2}（7.2.8；规范要求，按最小配筋作为提示列出）"),
    (r"f'cu = (\S+) MPa < 0\.8 × C(\d+) = (\S+) MPa when prestressing \(7\.2\.3; required by the code, a material "
     r"rule reported as a warning\)",
     "施加预应力时 f′cu = {1} MPa < 0.8 × C{2} = {3} MPa（7.2.3；规范要求，作为材料提示列出）"),
    (r"E'c = (\S+) MPa < 0\.8 Ec\(28 d\) = (\S+) MPa when prestressing \(7\.2\.3; required by the code, a material "
     r"rule reported as a warning\)",
     "施加预应力时 E′c = {1} MPa < 0.8Ec（28 d）= {2} MPa（7.2.3；规范要求，作为材料提示列出）"),
    (r"5\.2\.13-1 with ξ = (\S+)", "按式（5.2.13-1），ξ = {1}"),
    (r"prestressed: commentary to 5\.2\.13, formula 5-9 \(the stirrups carry γ0Vd − Vpb; ξ not used\)",
     "预应力构件：按 5.2.13 条文说明式（5-9），箍筋承担 γ0Vd − Vpb，不用 ξ"),
    (r"spacing (\S+) mm: within one beam depth \((\S+) mm\) of the support centre at most 100 mm \(9\.3\.12 item 3\)",
     "箍筋间距 {1} mm：支座中心向跨径方向一倍梁高（{2} mm）范围内不宜大于 100 mm（9.3.12 第 3 款）"),
    (r"stirrup diameter (\S+) mm < 1/4 of the main bars \((\S+) mm\) \(9\.3\.12 item 1\)",
     "箍筋直径 {1} mm < 主筋直径（{2} mm）的 1/4（9.3.12 第 1 款）"),
    (r"spacing (\S+) mm > (\S+) mm \(9\.3\.12\)", "箍筋间距 {1} mm > {2} mm（9.3.12）"),
    (r"9\.3\.13 items 1, 2 not checked for: (.*)", "9.3.13 第 1、2 款未检查的项目：{c}"),
    (r"circle: resultant moment √\(Mdx² \+ Mdy²\) \(5\.3\.11 not needed; program convention\)",
     "圆形截面：取合弯矩 √(Mdx² + Mdy²)（不需按 5.3.11；程序约定）"),
    (r"3362 has no shear rule for circular sections: b, h0 of GB 50010-2010 6\.3\.15 \(program convention\)",
     "JTG 3362 无圆形截面抗剪规定：b、h0 按 GB 50010—2010 第 6.3.15 条取（程序约定）"),
    (r"axial compression not counted \(5\.2\.9 has no such term; safe side\)", "不计轴压力（5.2.9 无此项，偏安全）"),
    (r"axial tension (\S+) kN: Vcs − 0\.2 γ0 N = (\S+) kN, not below fsv Asv h0/s = (\S+) kN \(GB 50010 6\.3\.14, "
     r"program convention\)",
     "轴向拉力 {1} kN：Vcs − 0.2γ0N = {2} kN，不小于 fsv·Asv·h0/s = {3} kN（GB 50010 第 6.3.14 条，程序约定）"),
    (r"via jtg\.shear: column detailing, no axial force; piles and members in tension: jtg\.shear_circle\(\.\.\., "
     r"pile=True, Nd=…\)",
     "按墩柱构造、不计轴力计算；桩和受拉构件应另按计入轴力的圆形截面抗剪计算（程序约定）"),
    (r"9\.6\.3 side bars: the bars go all round a circle", "9.6.3 侧面钢筋：圆形截面纵筋沿周边布置"),
    (r"l/h ≤ 5: a higher concrete class is advised \(9\.6\.5\)", "l/h ≤ 5：宜提高混凝土强度等级（9.6.5）"),
    (r"piles at most 3 D apart: main bars evenly over the whole width \(9\.6\.10 item 2\)",
     "桩中距不大于 3D：主筋沿承台全宽均匀布置（9.6.10 第 2 款）"),
    (r"piles 3 D or more apart: hangers between the piles advised \(9\.6\.10 item 6\)",
     "桩中距不小于 3D：桩间宜设吊筋（9.6.10 第 6 款）"),
    (r"pc_class not given: As ≥ 0\.003 b h0 of partially prestressed members not checked \(9\.1\.13\)",
     "未给出预应力构件类别：部分预应力构件 As ≥ 0.003bh0 未检查（9.1.13）"),
    (r"pc_class not given: α2 = 1\.0 in 5\.5\.3-2", "未给出预应力构件类别：式（5.5.3-2）取 α2 = 1.0"),
    (r"As = (\S+) mm² < 0\.003 b h0 = (\S+) mm² \(9\.1\.13\)", "As = {1} mm² < 0.003bh0 = {2} mm²（9.1.13）"),
    (r"5\.2\.7: if the tension steel beyond the ultimate-limit-state need was added for serviceability or by "
     r"detailing, pass uls_steel_only=True \(Mu,b = (\S+) kN·m with the tension steel cut to x = ξb h0\)",
     "5.2.7：多出的受拉钢筋若为正常使用或构造所加，可只计承载能力所需的钢筋（受拉钢筋减至 x = ξbh0 时 Mu,b = {1} kN·m）"),
    (r"5\.2\.7: only the tension steel the ultimate limit state needs counted for x: (\S+) of As, Ap gives x = (\S+) "
     r"mm ≤ ξb h0; resistance Mu,b = (\S+) kN·m \(tension steel cut to x = ξb h0\)",
     "5.2.7：求 x 只计承载能力所需的受拉钢筋：取 As、Ap 的 {1} 倍得 x = {2} mm ≤ ξbh0；承载力取 Mu,b = {3} kN·m（受拉钢筋减至 x = ξbh0）"),
    (r"the steel provided is still over-reinforced: the failure is the more brittle one \(commentary to "
     r"5\.2\.1–5\.2\.7, p175\)",
     "实配钢筋仍超筋，破坏形态较脆（5.2.1～5.2.7 条文说明，p175）"),
    (r"over-reinforced: x = (\S+) > ξb h0 = (\S+) mm", "超筋：x = {1} > ξbh0 = {2} mm（5.2.2-3）"),
    (r"over-reinforced: x = (\S+) > ξb h0 = (\S+) mm; with only the steel the ultimate limit state needs \(5\.2\.7\) "
     r"still over-reinforced: γ0Md > Mu,b = (\S+) kN·m",
     "超筋：x = {1} > ξbh0 = {2} mm；只计承载能力所需钢筋（5.2.7）仍超筋：γ0Md > Mu,b = {3} kN·m"),
    (r"no flexural resistance", "无抗弯承载力"),
    (r"section too small: γ0Vd > 0\.51e-3 √fcu,k b h0 = (\S+) kN \(5\.2\.11\)",
     "截面尺寸不足：γ0Vd > 0.51×10⁻³√fcu,k bh0 = {1} kN（5.2.11）"),
    (r"e0 = (\S+) mm raised to the minimum (\S+) mm \(5\.3\.9\)", "e0 = {1} mm 小于下限，取 {2} mm（5.3.9）"),
    (r"(tension side|compression side) reinforcement (\S+) < minimum 0\.2 % \(9\.1\.12\)",
     "{s}一侧钢筋配筋率 {2} < 0.2%（9.1.12）"),
    (r"longitudinal reinforcement (\S+) < minimum (\S+) \(9\.1\.12\)", "全部纵向钢筋配筋率 {1} < 最小配筋率 {2}（9.1.12）"),
    (r"9\.1\.12 \(reinforced concrete members\) not applied to a prestressed section", "9.1.12（钢筋混凝土构件）不用于预应力截面"),
    (r"load between the gross and the plastic centroid: the far face is the more compressed",
     "轴力作用在毛截面重心与塑性中心之间：远离轴力的一侧受压较大"),
    (r"reverse failure governs \(e' = (\S+) mm, 5\.3\.4-4\)", "反向破坏控制（e′ = {1} mm，5.3.4-4）"),
    (r"bars at (\S+) MPa instead of fsd = (\S+) MPa \(table 3\.2\.3-1 note 1\)",
     "钢筋强度取 {1} MPa，不取 fsd = {2} MPa（表 3.2.3-1 注 1）"),
    (r"bars at (\S+) MPa instead of fsd = (\S+) MPa \(table 3\.2\.3-1 note 1\) in Nud; Mud with fsd",
     "Nud 中钢筋强度取 {1} MPa，不取 fsd = {2} MPa（表 3.2.3-1 注 1）；Mud 用 fsd"),
    (r"(.*): (tension face|far face) reinforcement (\S+) of the gross area < (\S+) \(9\.1\.12\)",
     "{1}：{2}钢筋配筋率 {3}（按全截面）< {4}（9.1.12）"),
    (r"(.*): half the bars (\S+) of the gross area < (\S+) \(9\.1\.12\)", "{1}：半数纵筋配筋率 {2}（按全截面）< {3}（9.1.12）"),
    (r"hollow circle: 5\.4\.4 is written for solid circles, applied here with Mud from the strain method \(program "
     r"convention\)",
     "空心圆截面：5.4.4 按实心圆写出，此处用应变法求 Mud（程序约定）"),
    (r"C(\d+) below C(\d+) for environment (\S+), (\w+), (\d+) years \(table 4\.5\.3\)",
     "混凝土 C{1} 低于 C{2}（{3} 类环境、{4}、设计使用年限 {5} 年，表 4.5.3）"),
    (r"ζ = (\S+) > 1\.7: taken as 1\.7 \(5\.5\.1\)", "ζ = {1} > 1.7，取 1.7（5.5.1）"),
    (r"ζ = (\S+) < 0\.6: too little longitudinal torsion steel for the stirrups \(5\.5\.1\)",
     "ζ = {1} < 0.6：抗扭纵筋相对箍筋过少（5.5.1）"),
    (r"prestress term 0\.05 Np0/A0 Wt counted \(ep0 ≤ h/6, ζ ≥ 1\.7\)", "计入预加力项 0.05Np0/A0·Wt（ep0 ≤ h/6，ζ ≥ 1.7）"),
    (r"prestress not counted \(ep0 = (\S+) mm vs h/6 = (\S+) mm, ζ = (\S+)\): as reinforced concrete \(5\.5\.1\)",
     "不计预加力项（ep0 = {1} mm，h/6 = {2} mm，ζ = {3}）：按钢筋混凝土计算（5.5.1）"),
    (r"shear with torsion: γ0Vd = (\S+) > Vu = (\S+) kN with all the stirrups \(5\.5\.4-1\)",
     "剪扭构件抗剪：γ0Vd = {1} > 全部箍筋的 Vu = {2} kN（5.5.4-1）"),
    (r"section too small: γ0Vd/\(bh0\) \+ γ0Td/Wt = (\S+) > 0\.51 √fcu,k = (\S+) MPa \(5\.5\.3-1\)",
     "截面尺寸不足：γ0Vd/(bh0) + γ0Td/Wt = {1} > 0.51√fcu,k = {2} MPa（5.5.3-1）"),
    (r"γ0Vd/\(bh0\) \+ γ0Td/Wt = (\S+) ≤ 0\.50 α2 ftd = (\S+) MPa: torsion detailing reinforcement suffices "
     r"\(5\.5\.3-2, 9\.3\.13\)",
     "γ0Vd/(bh0) + γ0Td/Wt = {1} ≤ 0.50α2ftd = {2} MPa：可按构造配置抗扭钢筋（5.5.3-2、9.3.13）"),
    (r"box: one cell assumed, webs (\S+) mm each \(pass t_web otherwise\)", "箱形截面：按单箱单室，每道腹板厚 {1} mm"),
    (r"(top|bottom) flange overhang (\S+) mm counted as 6 hf = (\S+) mm \(5\.5\.5\)",
     "{1}翼缘悬出 {2} mm，按 6hf = {3} mm 计（5.5.5）"),
    (r"cantilevers (\S+) mm counted as 6 hf = (\S+) mm \(5\.5\.5\)", "悬臂 {1} mm，按 6hf = {2} mm 计（5.5.5）"),
    (r"prestressed: 9\.3\.13 item 4 checks the torsion minimum alone \(the flexural minimum is 9\.1\.13\)",
     "预应力构件：9.3.13 第 4 款只查抗扭最小配筋（受弯最小配筋按 9.1.13）"),
    (r"stirrup ratio (\S+) < (\S+) \(9\.3\.13 item 3\)", "配箍率 {1} < {2}（9.3.13 第 3 款）"),
    (r"longitudinal torsion steel Ast/\(bh\) = (\S+) < (\S+) \(9\.3\.13 item 4\)",
     "抗扭纵筋 Ast/(bh) = {1} < {2}（9.3.13 第 4 款）"),
    (r"longitudinal steel As \+ Ast = (\S+) mm² < (\S+) mm²: the flexural minimum (\S+) b h0 plus the torsion "
     r"minimum (\S+) b h \(9\.3\.13 item 4\)",
     "纵筋 As + Ast = {1} mm² < {2} mm²：受弯最小 {3}bh0 加抗扭最小 {4}bh（9.3.13 第 4 款）"),
    (r"tendon stress (\S+) > (\S+) fpk = (\S+) MPa \(7\.1\.5-(\d)\)", "预应力钢筋应力 {1} > {2}fpk = {3} MPa（7.1.5-{4}）"),
    (r"external tendon stress σpe,ex = (\S+) > 0\.60 fpk = (\S+) MPa \(7\.1\.5-3\)",
     "体外束应力 σpe,ex = {1} > 0.60fpk = {2} MPa（7.1.5-3）"),
    (r"Np0 with Mk ± Mp2 leaves the whole section in compression \(no crack\): the larger edge compression is checked",
     "Np0 与 Mk ± Mp2 共同作用下全截面受压（不开裂）：取上、下缘较大的压应力验算"),
    (r"bar stress (\S+) > 0\.75 fsk = (\S+) MPa \(7\.2\.4-2\)", "钢筋应力 {1} > 0.75fsk = {2} MPa（7.2.4-2）"),
    (r"σttp = (\S+) > f'tk = (\S+) MPa \(7\.2\.5\)", "σttp = {1} > f′tk = {2} MPa（7.2.5）"),
    (r"σttp = (\S+) ≤ 0\.25 f'tk: carried by the concrete, detailing stirrups \(7\.2\.6-1\)",
     "σttp = {1} ≤ 0.25f′tk：由混凝土承担，箍筋按构造设置（7.2.6-1）"),
    (r"σttp = (\S+) > 0\.25 f'tk: stirrups and bent bars carry the shear stress \(7\.2\.6; along the member: "
     r"transient_shear\)",
     "σttp = {1} > 0.25f′tk：剪应力由箍筋和弯起钢筋承担（7.2.6；沿梁另行计算）"),
    (r"bent bars Asb ≥ b Ω / \(√2 \[σs\]\) = (\S+) mm² over a = (\S+) mm \(7\.2\.6-3; pass Asb to check\)",
     "弯起钢筋 Asb ≥ bΩ/(√2[σs]) = {1} mm²，范围 a = {2} mm（7.2.6-3；未给出 Asb，未检查）"),
    (r"bent bars Asb = (\S+) < b Ω / \(√2 \[σs\]\) = (\S+) mm² \(7\.2\.6-3\)",
     "弯起钢筋 Asb = {1} < bΩ/(√2[σs]) = {2} mm²（7.2.6-3）"),
    (r"5\.7 covers members with indirect reinforcement \(mesh or spiral\): none given; the resistance shown is "
     r"5\.7\.2-1 with ρv = 0",
     "5.7 适用于配置间接钢筋（方格网或螺旋筋）的构件，本处未配：所列抗力为式（5.7.2-1）取 ρv = 0（程序约定，仅供参考）"),
    (r"Acor = (\S+) > Ab: Acor = Ab \(5\.7\.2\)", "Acor = {1} > Ab：取 Acor = Ab（5.7.2）"),
    (r"mesh: bar areas of the two directions differ by more than 50 % \(5\.7\.2\)", "方格网：两个方向钢筋面积相差超过 50%（5.7.2）"),
    (r"(\d+) (mesh layers|spiral turns) < 4 \(5\.7\.2 note\)", "{1} {m} < 4（5.7.2 注）"),
    (r"te = (\S+) mm < (la|d)/10 = (\S+) mm", "te = {1} mm < {2}/10 = {3} mm（8.7.3-6）"),
    (r"shape factor S = (\S+) outside (\S+) … (\S+) \(JT/T 4-2019 5\.4\.7\): Ee = 5\.4 G S² does not apply, pass E",
     "形状系数 S = {1} 不在 {2}～{3} 范围内（JT/T 4—2019 第 5.4.7 条）：Ee = 5.4GS² 不适用，应给出 E"),
    (r"shape factor S = (\S+) outside (\S+) … (\S+) \(JT/T 4-2019 5\.4\.7\)",
     "形状系数 S = {1} 不在 {2}～{3} 范围内（JT/T 4—2019 第 5.4.7 条）"),
    (r"δc,m = (\S+) mm < θ la/2 = (\S+) mm: the bearing lifts off at one edge",
     "δc,m = {1} mm < θla/2 = {2} mm：支座一侧脱空"),
    (r"σc = (\S+) MPa above (\S+) MPa of JT/T 4-2019 5\.4\.1 \(S = (\S+)\)",
     "σc = {1} MPa 超过 JT/T 4—2019 第 5.4.1 条的 {2} MPa（S = {3}）"),
    (r"Rck ≥ 5000 kN: outside JT/T 4-2019 \(bearings below 5000 kN\), whose σc, Ee, Eb, μ are used",
     "Rck ≥ 5000 kN：超出 JT/T 4—2019 的适用范围（5000 kN 以下），所用 σc、Ee、Eb、μ 取自该标准"),
    (r"plates (\S+) mm < (\S+) mm \(8\.7\.3 item 4\)", "加劲钢板 {1} mm < {2} mm（8.7.3 第 4 款）"),
    (r"side cover of the plates (\S+) mm < (\S+) mm \(8\.7\.3 item 4\)", "钢板侧面保护层 {1} mm < {2} mm（8.7.3 第 4 款）"),
    (r"top / bottom cover (\S+) mm < (\S+) mm \(8\.7\.3 item 4\)", "上、下保护层 {1} mm < {2} mm（8.7.3 第 4 款）"),
    (r"rubber layers (\S+) mm < (\S+) mm between plates \(JT/T 4-2019 5\.3\.1\.2\)",
     "钢板间橡胶层 {1} mm < {2} mm（JT/T 4—2019 第 5.3.1.2 条）"),
    (r"soffit grade (\S+) > 1 %(.*)", "梁底纵坡 {1} > 1%（8.7.3；应调平支承垫石，9.7.4）"),
    (r"cross slope (\S+) > 2 %(.*)", "横坡 {1} > 2%（8.7.3 只适用于横坡不大于 2% 的支座）"),
    (r"β = (\S+) < 1\.2 \(8\.8\.2\)", "β = {1} < 1.2（8.8.2）"),
    (r"β = (\S+) above the 1\.2 … 1\.4 of 8\.8\.2", "β = {1} 超出 8.8.2 的 1.2～1.4"),
    (r"joint type '(\S+)': 9\.7\.7 names modular, finger and seamless joints; others after special study",
     "伸缩装置类型 '{1}'：9.7.7 列出模数式、梳齿板式和无缝式，其他类型需专门研究"),
    (r"no shrinkage \(steel bridge, no concrete given\)", "不计收缩（钢桥，未给出混凝土）"),
    (r"joint steel (\S+) mm² < 0\.3 b s / fsd = (\S+) mm² \(8\.1\.7-3\)",
     "叠合面钢筋 {1} mm² < 0.3bs/fsd = {2} mm²（8.1.7-3）"),
    (r"weld length (\S+) mm < (\S+) mm \((.*)\)", "焊缝长度 {1} mm < {2} mm（{c3}）"),
    (r"area of the splice bars (\S+) mm² < (\S+) mm² \((.*)\)", "帮条钢筋面积 {1} mm² < {2} mm²（{c3}）"),
    (r"two joints of one bar (\S+) mm apart, within l = (\S+) mm \((.*)\)", "同一钢筋两接头相距 {1} mm，在 l = {2} mm 范围内（{c3}）"),
    (r"share of welded bars in the zone (\S+) > (\S+) \((.*)\)", "区段内焊接接头钢筋比例 {1} > {2}（{c3}）"),
    (r"clear distance at the welded joint (\S+) mm < (\S+) mm \((.*)\)", "焊接接头处钢筋净距 {1} mm < {2} mm（{c3}）"),
    (r"couplers per JGJ 107-2016, JG/T 163-2013 and SL 677-2014 \(9\.1\.10, 9\.1\.11\)",
     "连接件按 JGJ 107—2016、JG/T 163—2013、SL 677—2014（9.1.10、9.1.11）"),
    (r"cover of the couplers against table 9\.1\.1 (\S+) mm < (\S+) mm \((.*)\)",
     "连接件保护层 {1} mm < 表 9.1.1 的 {2} mm（{c3}）"),
    (r"crossing tension bars at a re-entrant corner: each anchored la beyond the crossing, into the compression zone "
     r"\(9\.3\.14, not checked by numbers\)",
     "凹角处交叉的受拉钢筋：各自越过交点伸入受压区锚固 la（9.3.14，不作数值检查）"),
    (r"bottom bars through the support (\S+) < (\S+) \(9\.3\.9\)", "伸过支座的底部钢筋 {1} 根 < {2}（9.3.9）"),
    (r"webs under torsion: their mean width also by the note to 5\.5\.5 \(9\.3\.2 item 3\)",
     "受扭腹板：其平均宽度另按 5.5.5 注检查（9.3.2 第 3 款）"),
    (r"free edges need a band of ≥ 3 main bars, top and bottom \(9\.2\.6 item 1\)",
     "自由边须设不少于 3 根主筋的加强带，上下各一（9.2.6 第 1 款）"),
    (r"monolithic skew slab of (\S+)° > 15° with the main bars along the bridge axis: normal to the support axis "
     r"\(9\.2\.6 item 1, should\)",
     "整体式斜板斜交角 {1}° > 15°、主筋沿桥轴线：宜垂直于支承轴线布置（9.2.6 第 1 款，宜）"),
    (r"hinge opening for a poker vibrator; bars from the precast slabs into the hinge \(9\.2\.8\)",
     "铰缝开口应能插入振捣棒；预制板钢筋伸入铰缝（9.2.8）"),
    (r"CR rubber serves (\S+) … \+?(\S+) °C, site (\S+) … (\S+) °C \(9\.7\.2\)",
     "CR 橡胶适用 {1}～{2} °C，桥位 {3}～{4} °C（9.7.2）"),
    (r"(\S+) rubber serves (\S+) … \+?(\S+) °C, site (\S+) … (\S+) °C \(9\.7\.2\)",
     "{1} 橡胶适用 {2}～{3} °C，桥位 {4}～{5} °C（9.7.2）"),
    (r"precast culvert members: check them for handling and erection \(9\.8\.1\)", "预制涵洞构件：应验算吊运和安装（9.8.1）"),
    (r"round culvert of (\S+) m with (\d+) layer of bars: two layers from 1 m up \(9\.8\.1\)",
     "直径 {1} m 的圆管涵配 {2} 层钢筋：直径 1 m 及以上应配两层（9.8.1）"),
    (r"mesh ratio (\S+) < 0\.3 % \(B\.3\.5\)", "钢筋网配筋率 {1} < 0.3%（B.3.5）"),
    (r"mesh spacing (\S+) mm > 300 mm \(B\.3\.5\)", "钢筋网间距 {1} mm > 300 mm（B.3.5）"),
    (r"cover of the main bars(?: at the (bottom|top))? (\S+) mm > 50 mm.*\(9\.1\.2, should\)",
     "{fa}主筋保护层 {2} mm > 50 mm，宜在保护层内设钢筋网（9.1.2，宜）"),
]

_SKIP = "未给出输入、未检查："


def message_zh(s: str) -> str | None:
    """the Chinese of a program message, or None"""
    mt = re.fullmatch(r"skipped \(no input\): (.*)", s)
    if mt:
        items = []
        for it in mt.group(1).split("; "):
            m2 = re.fullmatch(r"(.*?) \(([^()]*)\)", it)
            name, cl = (m2.group(1), m2.group(2)) if m2 else (it, "")
            zh = rule_zh(name) or name
            items.append(f"{zh}（{clause(cl)}）" if cl else zh)
        return _SKIP + "；".join(items)
    for pat, zh in MESSAGES:
        if zh is None:
            continue
        mt = re.fullmatch(pat, s)
        if not mt:
            continue
        out = zh
        if "{b}" in out:
            out = out.replace("{b}", "下" if mt.group(1) == "below" else "上")
        if "{s}" in out:
            out = out.replace("{s}", "受拉" if mt.group(1).startswith("tension") else "受压")
        if "{m}" in out:
            out = out.replace("{m}", "层钢筋网" if mt.group(2) == "mesh layers" else "圈螺旋筋")
        if "{fa}" in out:
            out = out.replace("{fa}", {"bottom": "底面", "top": "顶面"}.get(mt.group(1) or "", ""))
        if "{c}" in out:
            out = out.replace("{c}", "、".join({"closed": "封闭", "hook": "弯钩", "bar_spacing": "纵筋间距",
                                               "corner_bars": "角部纵筋"}.get(x, x) for x in mt.group(1).split(", ")))
        out = re.sub(r"\{c(\d+)\}", lambda k: clause(mt.group(int(k.group(1)))), out)
        return re.sub(r"\{(\d+)\}", lambda k: word(mt.group(int(k.group(1))) or ""), out)
    # a rule: "<rule> <value>[ unit] <|> <limit>[ unit] (<clause>)"
    mt = re.fullmatch(r"(.+?) (−?-?[\d.]+%?(?: mm²?| MPa)?) ([<>]) (\S+(?: mm²?| MPa)?) \(([^()]*)\)", s)
    if mt:
        rule = rule_zh(mt.group(1))
        if rule is not None:
            sep = "" if rule[-1] in "）》】" else " "
            return f"{rule}{sep}{mt.group(2)} {mt.group(3)} {mt.group(4)}（{clause(mt.group(5))}）"
    return None


# ── values of Check.details (the general form) ──────────────────────────────
# key → (name, unit); a name in the math language between $ … $ is a symbol
KEYS = {
    "x": ("受压区高度 $x$", "mm"), "h0": ("有效高度 $h_{0}$", "mm"), "xi": ("$ξ$", ""), "xi_b": ("$ξ_{b}$", ""),
    "b": ("宽度 $b$", "mm"), "rho": ("配筋率 $ρ$", ""), "rho_min": ("最小配筋率 $ρ_{min}$", ""), "P": ("配筋百分率 $P$", ""),
    "rho_sv": ("配箍率 $ρ_{sv}$", ""), "alpha1": ("$α_{1}$", ""), "alpha2": ("$α_{2}$", ""), "alpha3": ("$α_{3}$", ""),
    "Vcs": ("$V_{cs}$", "kN"), "Vsb": ("$V_{sb}$", "kN"), "Vpb": ("$V_{pb}$", "kN"), "Vpb_ex": ("$V_{pb,ex}$", "kN"),
    "fsv": ("$f_{sv}$", "MPa"), "sigma_ss": ("钢筋应力 $σ_{ss}$", "MPa"), "C1": ("$C_{1}$", ""), "C2": ("$C_{2}$", ""),
    "C3": ("$C_{3}$", ""), "c": ("保护层 $c$", "mm"), "d": ("钢筋直径 $d$", "mm"), "rho_te": ("$ρ_{te}$", ""),
    "rho_te_used": ("取用的 $ρ_{te}$", ""), "E": ("弹性模量 $E$", "MPa"), "W0": ("$W_{0}$", "mm³"),
    "e0": ("初始偏心距 $e_{0}$", "mm"), "eta": ("偏心距增大系数 $η$", ""), "e": ("偏心距 $e$", "mm"), "phi": ("稳定系数 $φ$", ""),
    "Mcr": ("开裂弯矩 $M_{cr}$", "kN·m"), "sigma_pc": ("预压应力 $σ_{pc}$", "MPa"), "sigma_st": ("$σ_{st}$", "MPa"),
    "sigma_lt": ("$σ_{lt}$", "MPa"), "Np0": ("$N_{p0}$", "kN"), "beta": ("$β$", ""), "Ab": ("$A_{b}$", "mm²"),
    "Al": ("$A_{l}$", "mm²"), "Aln": ("$A_{ln}$", "mm²"), "fcd": ("$f_{cd}$", "MPa"), "fsd": ("$f_{sd}$", "MPa"),
    "ftd": ("$f_{td}$", "MPa"), "ftk": ("$f_{tk}$", "MPa"), "Um": ("$U_{m}$", "mm"), "z": ("内力臂 $z$", "mm"),
    "Ate": ("$A_{te}$", "mm²"), "As": ("$A_{s}$", "mm²"), "Ap": ("$A_{p}$", "mm²"), "Asv": ("$A_{sv}$", "mm²"),
    "sv": ("箍筋间距 $s_{v}$", "mm"), "a_s": ("$a_{s}$", "mm"), "Ms": ("$M_{s}$", "kN·m"), "Ml": ("$M_{l}$", "kN·m"),
    "N": ("轴力 $N$", "kN"), "Mk": ("$M_{k}$", "kN·m"), "l0": ("计算长度 $l_{0}$", "mm"), "Nd": ("$N_{d}$", "kN"),
    "Md": ("$M_{d}$", "kN·m"), "Wt": ("$W_{t}$", "mm³"), "Acor": ("$A_{cor}$", "mm²"), "zeta": ("$ζ$", ""),
    "zeta_used": ("取用的 $ζ$", ""), "beta_t": ("$β_{t}$", ""),
    "V_section_max": ("$0.51×10^{-3}\\sqrt{f_{cu,k}}bh_{0}$", "kN"),
    "V_detailing": ("$0.50×10^{-3}α_{2}f_{td}bh_{0}$", "kN"), "L": ("计算跨径 $L$", "mm"), "te": ("橡胶层总厚度 $t_{e}$", "mm"),
    "Ae": ("$A_{e}$", "mm²"), "Rck": ("$R_{ck}$", "kN"), "A": ("截面面积 $A$", "mm²"), "A_used": ("取用面积 $A$", "mm²"),
    "Acr": ("$A_{cr}$", "mm²"), "Icr": ("$I_{cr}$", "mm⁴"), "I0": ("$I_{0}$", "mm⁴"), "x0": ("受压区高度 $x_{0}$", "mm"),
    "eN": ("$e_{N}$", "mm"), "e0N": ("$e_{0N}$", "mm"), "hps": ("$h_{ps}$", "mm"), "sigma_p": ("钢筋应力增量 $σ_{p}$", "MPa"),
    "tendon_stress": ("预应力钢筋应力", "MPa"), "tendon_limit": ("预应力钢筋应力限值", "MPa"), "edge": ("控制边缘", ""),
    "edge_stress": ("边缘应力", "MPa"), "sigma_kc": ("$σ_{kc}$", "MPa"), "sigma_pt": ("$σ_{pt}$", "MPa"),
    "sigma_kt": ("$σ_{kt}$", "MPa"), "M_G": ("自重弯矩 $M_{G}$", "kN·m"), "M_p2": ("预加力次弯矩 $M_{p2}$", "kN·m"),
    "sigma_G": ("$σ_{G}$", "MPa"), "sigma_t": ("温差应力 $σ_{t}$", "MPa"), "points": ("计算点", ""), "at": ("控制点", ""),
    "gamma": ("$γ$", ""), "cracking": ("开裂弯矩计算", ""), "b_te": ("受拉翼缘宽 $b$", "mm"), "As_te": ("$A_{s}$", "mm²"),
    "Nl": ("$N_{l}$", "kN"), "member": ("受力性质", ""), "e0_from_forces": ("$M_{d}/N_{d}$", "mm"),
    "N_reverse": ("反向破坏承载力", "kN"), "case": ("偏心类型", ""), "a_ref": ("受压钢筋合力点 $a′$", "mm"),
    "M_about_compression_steel": ("对受压钢筋合力点取矩 $M_{u1}$", "kN·m"), "M_singly_reinforced": ("单筋截面 $M_{u2}$", "kN·m"),
    "clause_5_2_7": ("5.2.7 的处理", ""), "k_balanced": ("受拉钢筋界限比例 $k_{b}$", ""), "x_balanced": ("$x_{b}$", "mm"),
    "Mu_balanced": ("$M_{u,b}$", "kN·m"), "k_uls": ("承载所需受拉钢筋比例", ""), "x_uls": ("承载所需钢筋的 $x$", "mm"),
    "N_ex": ("体外束拉力", "kN"), "Nud": ("$N_{ud}$", "kN"), "fsd_tension": ("钢筋抗拉强度取值", "MPa"), "e_prime": ("$e′$", "mm"),
    "N_5_4_2_1": ("式（5.4.2-1）承载力", "kN"), "N_5_4_2_2": ("式（5.4.2-2）承载力", "kN"), "rho_tension": ("受拉面配筋率", ""),
    "rho_far": ("远离轴力一侧配筋率", ""), "rho_total": ("全部纵筋配筋率", ""), "rho_compression": ("受压一侧配筋率", ""),
    "moduli": ("受扭塑性抵抗矩", ""), "T_share": ("分配扭矩", "kN·m"), "Vu": ("$V_{u}$", "kN"), "shear_ratio": ("抗剪利用率", ""),
    "rho_sv_shear": ("抗剪所需配箍率", ""), "Ucor": ("$U_{cor}$", "mm"), "Asv1": ("$A_{sv1}$", "mm²"),
    "ep0": ("$e_{p0}$", "mm"), "prestress_counted": ("计入预加力项", ""), "beta_a": ("$β_{a}$", ""),
    "q_provided": ("实配 $A_{sv1}/s_{v}$", "mm²/mm"), "q_shear": ("抗剪所需 $A_{sv1}/s_{v}$", "mm²/mm"),
    "q_torsion": ("抗扭可用 $A_{sv1}/s_{v}$", "mm²/mm"), "tau_section": ("$γ_{0}V_{d}/(bh_{0}) + γ_{0}T_{d}/W_{t}$", "MPa"),
    "f_cv": ("$0.51\\sqrt{f_{cu,k}}$", "MPa"), "rho_sv_min": ("最小配箍率", ""), "rho_st": ("抗扭纵筋配筋率", ""),
    "rho_st_min": ("抗扭纵筋最小配筋率", ""), "As_flexure": ("受弯受拉钢筋 $A_{s}$", "mm²"), "rho_flexure_min": ("受弯最小配筋率", ""),
    "long_provided": ("实配纵筋", "mm²"), "long_required": ("所需纵筋", "mm²"), "fck_t": ("$f′_{ck}$", "MPa"),
    "ftk_t": ("$f′_{tk}$", "MPa"), "M": ("弯矩 $M$", "kN·m"), "sigma_s": ("钢筋应力", "MPa"), "dynamic_factor": ("动力系数", ""),
    "dynamic_factor_steel": ("动力系数（钢筋）", ""), "dynamic_factor_shear": ("动力系数（剪应力）", ""),
    "dynamic_factor_tension": ("动力系数（预拉区）", ""), "z0": ("$z_{0}$", "mm"), "sigma_tp": ("主拉应力 $σ_{tp}$", "MPa"),
    "V": ("剪力 $V$", "kN"), "tau_v": ("$τ_{v}$", "MPa"), "sigma_tct": ("$σ^{t}_{ct}$", "MPa"),
    "M_tension": ("预拉区弯矩", "kN·m"), "rho_pretension_zone": ("预拉区配筋率", ""), "rho_needed": ("所需配筋率", ""),
    "top": ("上缘应力", "MPa"), "bottom": ("下缘应力", "MPa"), "eta_s": ("$η_{s}$", ""), "F_section_max": ("$F_{u1}$", "kN"),
    "F_resistance": ("$F_{u2}$", "kN"), "rho_v": ("$ρ_{v}$", ""), "beta_cor": ("$β_{cor}$", ""), "k": ("$k$", ""),
    "governs": ("控制条件", ""), "beta_h": ("$β_{h}$", ""), "h0_min": ("最小有效高度", "mm"), "sigma_pc_m": ("$σ_{pc,m}$", "MPa"),
    "Asvu": ("$A_{svu}$", "mm²"), "Asbu": ("$A_{sbu}$", "mm²"), "theta": ("$θ$", "°"), "V_concrete": ("混凝土部分", "kN"),
    "V_steel": ("钢筋部分", "kN"), "V_outer": ("外圈抗冲切承载力", "kN"), "Fld": ("$F_{ld}$", "kN"), "bm": ("$b_{m}$", "mm"),
    "bb": ("$b_{b}$", "mm"), "ax": ("$a_{x}$", "mm"), "ay": ("$a_{y}$", "mm"), "lambda_x": ("$λ_{x}$", ""),
    "lambda_y": ("$λ_{y}$", ""), "alpha_px": ("$α_{px}$", ""), "alpha_py": ("$α_{py}$", ""), "m": ("$m$", ""),
    "tau": ("$τ$", "MPa"), "governing": ("控制情况", ""), "governing_tie": ("控制拉杆", ""), "a": ("$a$", "mm"),
    "y": ("$y$", "mm"), "n": ("$n$", ""), "T1": ("$T_{1}$", "kN"), "T2": ("$T_{2}$", "kN"), "S": ("形状系数 $S$", ""),
    "Ee": ("$E_{e}$", "MPa"), "Eb": ("$E_{b}$", "MPa"), "delta_cm": ("$δ_{c,m}$", "mm"),
    "uplift_limit": ("$θl_{a}/2$", "mm"), "la": ("$l_{a}$", "mm"), "fy": ("$f_{y}$", "MPa"), "Kp": ("$K_{p}$", ""),
    "tes_u_plus_l": ("$t_{es,u} + t_{es,l}$", "mm"), "sigma_s1": ("$σ_{s1}$", "MPa"), "sigma_s2": ("$σ_{s2}$", "MPa"),
    "dl": ("$Δ_{l}$", "mm"), "dt": ("$Δ_{t}$", "mm"), "dl_slope": ("坡度引起的 $Δ_{l}$", "mm"),
    "dl_brake": ("制动力引起的 $Δ_{l}$", "mm"), "tan_alpha": ("$\\op{tan}α$", ""), "te_min": ("$l_{a}/10$", "mm"),
    "te_max": ("$l_{a}/5$", "mm"), "Ae_required": ("所需 $A_{e}$", "mm²"), "sigma_c_jtt4": ("$σ_{c}$", "MPa"),
    "Ag": ("$A_{g}$", "mm²"), "Ge": ("$G_{e}$", "MPa"), "mu": ("$μ$", ""), "R": ("$R$", "kN"),
    "Fbk": ("$F_{bk}$", "kN"), "H_shear": ("$1.4G_{e}A_{g}Δ_{l}/t_{e}$", "kN"), "shear_angle": ("剪切角", ""),
    "C_plus": ("$C^{+}$", "mm"), "C_minus": ("$C^{-}$", "mm"), "t_plus": ("$Δl_{t}^{+}$", "mm"),
    "t_minus": ("$Δl_{t}^{-}$", "mm"), "shrinkage": ("$Δl_{s}^{-}$", "mm"), "creep": ("$Δl_{c}^{-}$", "mm"),
    "brake": ("$Δl_{b}$", "mm"), "installation": ("安装宽度", "mm"), "environment": ("环境类别", ""), "life": ("设计使用年限", "年"),
    "class": ("混凝土强度等级 C", ""), "min_class_4.5.3": ("表 4.5.3 要求的最低强度等级 C", ""), "cmin": ("最小保护层厚度 $c_{min}$", "mm"),
    "layers (bottom)": ("底排钢筋层数", ""), "layers (top)": ("顶排钢筋层数", ""), "spacing limit (bottom)": ("底排钢筋净距限值", "mm"),
    "spacing limit (top)": ("顶排钢筋净距限值", "mm"), "9.3.7 neutral axis": ("9.3.7 受拉区界线（距顶面）", "mm"),
    "transfer_factor": ("传递长度内的应力比例", ""), "psi_t": ("温差作用的 $ψ$", ""), "temperature": ("温差情况", ""),
    "cases": ("各情况", ""), "tendon_case": ("钢筋应力的控制情况", ""), "M_s": ("$M_{s}$", "kN·m"), "N_s": ("$N_{s}$", "kN"),
    "sigma_ss_load": ("荷载引起的 $σ_{ss}$", "MPa"), "sigma_s_t": ("温差引起的钢筋应力", "MPa"), "gamma_f": ("$γ′_{f}$", ""),
    "es": ("$e_{s}$", "mm"), "e's": ("$e′_{s}$", "mm"), "eta_s ": ("$η_{s}$", ""), "ep": ("$e_{p}$", "mm"),
    "N_total": ("合力", "kN"), "ys": ("$y_{s}$", "mm"), "external_stress": ("体外束应力", "MPa"),
    "external_limit": ("体外束应力限值", "MPa"), "l0_over_i": ("$l_{0}/i$", ""), "slenderness_by": ("长细比按", ""),
    "slenderness_dim": ("截面尺寸", "mm"), "N_5_3_1": ("式（5.3.1）承载力", "kN"), "N_5_3_2": ("式（5.3.2-1）承载力", "kN"),
    "Aso": ("$A_{so}$", "mm²"), "dcor": ("$d_{cor}$", "mm"), "N_5_3_11": ("式（5.3.11）承载力", "kN"),
    "Nu0": ("$N_{u0}$", "kN"), "Nux": ("$N_{ux}$", "kN"), "Nuy": ("$N_{uy}$", "kN"), "e0x": ("$e_{0x}$", "mm"),
    "e0y": ("$e_{0y}$", "mm"), "Mux": ("$M_{ux}$", "kN·m"), "Muy": ("$M_{uy}$", "kN·m"), "alpha": ("$α$", ""),
    "eta_e0_r": ("$ηe_{0}/r$", ""), "rs_over_r": ("$r_{s}/r$", ""), "rho_fsd_fcd": ("$ρf_{sd}/f_{cd}$", ""),
    "N_about_compression_steel": ("对受压钢筋合力点取矩的承载力", "kN"), "N_singly_reinforced": ("不计受压钢筋的承载力", "kN"),
    "N_without_compression_bars": ("不计受压钢筋的承载力", "kN"), "N_t": ("温差次轴力", "kN"), "M_k": ("$M_{k}$", "kN·m"),
    "sigma_t_edge": ("边缘温差应力", "MPa"), "V0d": ("$V_{0d}$", "kN"), "Vd_max_range": ("范围内最大剪力", "kN"),
    "b_min": ("最小宽度", "mm"), "V_stirrups": ("箍筋部分", "kN"), "V_tension": ("轴拉影响", "kN"), "bent": ("弯起钢筋", ""),
    "bent_tendons": ("弯起预应力钢筋", ""), "bent_external": ("弯起体外束", ""), "bent_crossed": ("相交的弯起钢筋", ""),
    "rho_pv": ("$ρ_{pv}$", ""), "fpv": ("$f_{pv}$", "MPa"), "rho_sv_provided": ("实配配箍率", ""),
    "rho_sv_mean": ("平均配箍率", ""), "legs": ("箍筋肢数", ""), "dia": ("箍筋直径", "mm"), "nu": ("$ν$", ""),
    "p_mean": ("平均压应力", "MPa"), "Omega": ("$Ω$", "N/mm"), "tau_max": ("$τ_{max}$", "MPa"),
    "Asb_needed": ("所需弯起钢筋", "mm²"), "x_a": ("$a$", "mm"), "h1_used": ("$h_{1}$", "mm"), "zone_l": ("区段长度", "mm"),
    "A_loop": ("吊环面积", "mm²"), "n_effective": ("有效吊环数", ""), "diameter": ("直径", "mm"), "layers": ("层数", ""),
    "d_max": ("最大直径", "mm"), "d_min": ("最小直径", "mm"), "spacing": ("间距", "mm"), "least_dimension": ("最小尺寸", "mm"),
    "rubber_range": ("橡胶适用温度", "°C"), "edge_strip": ("边带宽度", "mm"), "middle_x": ("中带宽度 x", "mm"),
    "middle_y": ("中带宽度 y", "mm"), "ratio_x": ("x 向配筋率", ""), "ratio_z": ("z 向配筋率", ""), "node_type": ("节点类型", ""),
    "confine": ("约束增强系数", ""), "eps1": ("$ε_{1}$", ""), "fce_d": ("$f_{ce,d}$", "MPa"), "beta_c": ("$β_{c}$", ""),
    "beta_n": ("$β_{n}$", ""), "theta_s": ("$θ_{s}$", "°"), "beyond_full_min": ("超过充分利用截面的最小长度", "mm"),
    "beyond_needed_min": ("超过不需要截面的最小长度", "mm"), "r": ("半径 $r$", "mm"), "M1u": ("$M_{1u}$", "kN·m"),
    "h_over_a": ("$h/a$", ""), "As_side": ("侧面钢筋", "mm²"), "Acs": ("$A_{cs}$", "mm²"), "An": ("$A_{n}$", "mm²"),
    "concrete_used": ("所用混凝土", ""), "h": ("$h$", "mm"), "ha": ("$h_{a}$", "mm"), "ftk ": ("$f_{tk}$", "MPa"),
    "l/h": ("$l/h$", ""), "l_h": ("$l/h$", ""), "ls_min": ("最小搭接长度", "mm"), "distribution_ratio": ("分配比例", ""),
    "worst_ratio": ("最大比值", ""), "spacing / D": ("间距 / 直径", ""), "mesh_area_ratio": ("两向钢筋面积比", ""),
    "curvature": ("曲率", "1/m"), "T": ("$T$", "kN·m"), "Cin": ("内侧 $C$", ""), "Cout": ("外侧 $C$", ""), "C": ("$C$", ""),
    "Ec": ("$E_{c}$", "MPa"), "W01": ("$W_{01}$", "mm³"), "x_c": ("$x_{c}$", "m"), "x_t": ("$x_{t}$", "m"),
    "xi_b_h0": ("$ξ_{b}h_{0}$", "mm"), "t": ("厚度 $t$", "mm"), "edges": ("各边缘", ""), "Vd": ("$V_{d}$", "kN"),
    "Td": ("$T_{d}$", "kN·m"), "Wt_total": ("全截面 $W_{t}$", "mm³"), "Ast": ("$A_{st}$", "mm²"),
    "gamma0": ("$γ_{0}$", ""), "A0": ("$A_{0}$", "mm²"), "alpha_ES": ("$α_{ES}$", ""), "h0i": ("各排钢筋 $h_{0i}$", "mm"),
    "Acor_indirect": ("间接钢筋内表面范围内的 $A_{cor}$", "mm²"), "Acor_raw": ("未截取的 $A_{cor}$", "mm²"),
    "Asv_min": ("最小箍筋面积", "mm²"), "At": ("$A_{p} + A_{s}$", "mm²"), "Icr_steel": ("钢筋应力用的 $I_{cr}$", "mm⁴"),
    "M_steel": ("钢筋应力用的弯矩", "kN·m"), "x0_steel": ("钢筋应力用的 $x_{0}$", "mm"), "Mt": ("$M_{k} ± M_{p2}$", "kN·m"),
    "Nt": ("温差次轴力", "kN"), "S0": ("$S_{0}$", "mm³"), "Um_outer": ("配筋区外的 $U_{m}$", "mm"), "alpha_EP": ("$α_{EP}$", ""),
    "angle": ("主应力方向角", "°"), "beta_t_shear": ("抗剪用的 $β_{t}$", ""), "body": ("腹板（主体）", ""), "box": ("箱形截面", ""),
    "bt": ("$b_{t}$", "mm"), "c_raw": ("实际保护层", "mm"), "cantilever": ("悬臂", ""), "capped": ("受限后的值", ""),
    "demand": ("作用效应", ""), "dims": ("尺寸", "mm"), "extra": ("其他应力", "MPa"), "fsk": ("$f_{sk}$", "MPa"),
    "hp": ("预应力钢筋深度 $h_{p}$", "mm"), "limit": ("限值", ""), "value": ("实际值", ""), "mesh": ("钢筋网", ""),
    "spiral": ("螺旋筋", ""), "mu_f": ("$μ_{f}$", ""), "negative": ("反温差", ""), "positive": ("正温差", ""),
    "none": ("不计温差", ""), "point": ("计算点", ""), "precast bottom": ("预制梁下缘", "MPa"), "precast top": ("预制梁上缘", "MPa"),
    "topping top": ("现浇层上缘", "MPa"), "ps": ("$p_{s}$", "MPa"), "ratio": ("比值", ""), "rule": ("规定", ""),
    "sigma_cp": ("$σ_{cp}$", "MPa"), "sigma_cx": ("$σ_{cx}$", "MPa"), "sigma_cy": ("$σ_{cy}$", "MPa"),
    "sigma_p0": ("$σ_{p0}$", "MPa"), "sigma_pe": ("$σ_{pe}$", "MPa"), "stirrup_fsd": ("箍筋 $f_{sd}$", "MPa"),
    "tendon_depth": ("预应力钢筋深度", "mm"), "total": ("合计", ""), "y0": ("$y_{0}$", "mm"), "fcuk": ("$f_{cu,k}$", "MPa"),
    "xi_b_alternative": ("另一种 ξb 读法", ""), "reading": ("读法", ""), "over_reinforced": ("超筋", ""), "a'": ("$a′$", "mm"),
    "Fbk_used": ("制动力", "kN"), "main_rows": ("主筋排数（底、顶）", "")
}
# keys that mean another thing in some checks: (start of the check's name, key) → (name, unit)
KEYS_IN = {
    ("crack width", "x_t"): ("温差计算的 $x$", "mm"), ("concrete compression", "x_t"): ("温差计算的 $x$", "mm")
}

VALUES = {
    "small eccentricity": "小偏心", "large eccentricity": "大偏心", "axial tension": "轴心受拉",
    "eccentric tension (circle)": "偏心受拉（圆形截面）", "superstructure": "上部结构", "substructure": "下部结构", "foundation": "基础",
    "flexural": "受弯", "compression": "受压", "tension": "受拉", "top": "上缘", "bottom": "下缘", "positive": "正温差",
    "negative": "反温差", "none": "不计", "5.7.1-1 (section size)": "5.7.1-1（截面尺寸）", "5.7.2-1": "5.7.2-1", "b": "宽度",
    "i": "回转半径", "d": "直径", "CCC": "CCC", "CCT": "CCT", "CTT": "CTT", "strain": "应变法", "code": "规范公式"
}
