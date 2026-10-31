#!/usr/bin/env python3
"""评分脚本：python3 grade.py <标准答案.xlsx> <候选人提交.xlsx>
按（标的, 序号）对齐；输出覆盖率、偏离类型准确率、★项关键召回、证明材料命中率、位置命中率。"""
import sys, re, json
from openpyxl import load_workbook

def norm(s):
    return re.sub(r"\s+", "", str(s or "")).replace("．", ".").strip()

def load(path):
    ws = load_workbook(path, data_only=True).active
    hdr = [norm(c.value) for c in ws[1]]
    def col(*names):
        for n in names:
            if n in hdr: return hdr.index(n)
        return None
    ci = dict(lot=col("标的"), sec=col("分节"), no=col("序号"), star=col("标记"), req=col("招标要求（原文）", "招标要求"),
              resp=col("投标响应", "响应值"), dev=col("偏离类型", "偏离"), file=col("证明材料文件", "证明材料"), loc=col("位置", "页码/位置", "页码"))
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r is None or ci["no"] is None or r[ci["no"]] in (None, ""): continue
        key = (norm(r[ci["lot"]]) if ci["lot"] is not None else "1", norm(r[ci["sec"]]) if ci["sec"] is not None else "", norm(r[ci["no"]]))
        rows[key] = {k: (norm(r[i]) if i is not None else "") for k, i in ci.items()}
    return rows

DEVN = {"无偏离": "无偏离", "正偏离": "正偏离", "负偏离": "负偏离", "未响应": "未响应", "响应": "无偏离", "符合": "无偏离", "满足": "无偏离", "优于": "正偏离", "不满足": "负偏离", "不符合": "负偏离", "缺失": "未响应", "未提及": "未响应", "无": "无偏离"}

def ndev(s):
    s = norm(s)
    for k, v in DEVN.items():
        if k in s: return v
    return s

def main():
    gold, sub = load(sys.argv[1]), load(sys.argv[2])
    gold = {k: v for k, v in gold.items() if v["dev"] != "不判分"}
    n = len(gold); hit = dev_ok = file_ok = loc_ok = 0
    star_total = star_ok = 0; conf = {}
    for k, g in gold.items():
        s = sub.get(k)
        gd = ndev(g["dev"])
        if s is None:
            conf[(gd, "缺行")] = conf.get((gd, "缺行"), 0) + 1
            if g["star"] in ("★", "▲") and gd in ("负偏离", "未响应"): star_total += 1
            continue
        hit += 1
        sd = ndev(s["dev"])
        conf[(gd, sd)] = conf.get((gd, sd), 0) + 1
        if sd == gd: dev_ok += 1
        if g["star"] in ("★", "▲") and gd in ("负偏离", "未响应"):
            star_total += 1
            if sd == gd: star_ok += 1
        if gd not in ("未响应",):
            gf = g["file"]; sf = s["file"]
            if gf and sf and (gf in sf or sf in gf or gf.split("_")[-1].split(".")[0] in sf): file_ok += 1
            gl, sl = g["loc"], s["loc"]
            gnum = re.findall(r"\d+", gl); snum = re.findall(r"\d+", sl)
            if gnum and snum and gnum[-1] in snum: loc_ok += 1
    extra = len([k for k in sub if k not in gold])
    judged = n - sum(1 for g in gold.values() if ndev(g["dev"]) == "未响应")
    out = dict(条目总数=n, 提交匹配条目=hit, 多余条目=extra,
               覆盖率=round(hit / n, 3) if n else 0,
               偏离类型准确率=round(dev_ok / n, 3) if n else 0,
               关键项风险识别=f"{star_ok}/{star_total}（★/▲且为负偏离或未响应的条目中被正确识别的数量）",
               证明材料命中率=round(file_ok / judged, 3) if judged else 0,
               位置命中率=round(loc_ok / judged, 3) if judged else 0,
               混淆矩阵={f"{a}->{b}": c for (a, b), c in sorted(conf.items())})
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
