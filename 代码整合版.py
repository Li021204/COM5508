
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


warnings.filterwarnings("ignore")


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _first_existing_dir(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


def _auto_detect_base_path(label: str) -> str:
    """
    在当前工程内自动探测数据目录，使得不传 --base-path 也尽量“开箱即跑”。
    注意：如果工程内根本没有数据文件，仍会在运行时提示缺少哪些文件。
    """
    base = _script_dir()

    # 常见数据文件夹命名（优先级从高到低）
    preferred = []
    if label == "10":
        preferred = [
            os.path.join(base, "10"),
            os.path.join(base, "data", "10"),
            os.path.join(base, "dataset", "10"),
            os.path.join(base, "5508 visual code", "10"),  # 当前工程已存在
        ]
    elif label == "11":
        preferred = [
            os.path.join(base, "11月数据_标签"),
            os.path.join(base, "data", "11月数据_标签"),
            os.path.join(base, "dataset", "11月数据_标签"),
            os.path.join(base, "5508 visual code", "11"),  # 当前工程已存在
        ]
    elif label == "12":
        preferred = [
            os.path.join(base, "12月标签"),
            os.path.join(base, "data", "12月标签"),
            os.path.join(base, "dataset", "12月标签"),
            os.path.join(base, "5508 visual code", "12"),  # 当前工程已存在
        ]

    detected = _first_existing_dir(preferred)
    return detected or base


def _default_base_path_for(label: str) -> str:
    """
    给出一个“相对当前目录”的默认数据路径。
    这些默认值不保证存在，但不会像原脚本一样绑死在别人的电脑路径上。
    """
    return _auto_detect_base_path(label)


def _auto_detect_trend_data_dir() -> str:
    """
    trend.py 用到的输入数据目录自动探测（可用 --data-dir 覆盖）。
    优先匹配常见的项目目录结构，找不到就回退到当前脚本目录。
    """
    base = _script_dir()
    preferred = [
        os.path.join(base, "data", "new_analysis"),
        os.path.join(base, "dataset", "new_analysis"),
        os.path.join(base, "new_analysis"),
        os.path.join(base, "data_2025_full", "new_analysis"),
    ]
    return _first_existing_dir(preferred) or base


def _require_path_exists(path: str, what: str = "路径") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what}不存在：{path}")


def _require_files(base_path: str, required_files: Iterable[str], *, what: str) -> None:
    missing = [f for f in required_files if not os.path.exists(os.path.join(base_path, f))]
    if missing:
        missing_list = "\n".join([f" - {m}" for m in missing])
        raise FileNotFoundError(
            f"{what}缺少必要数据文件（当前 base_path: {base_path}）：\n{missing_list}\n\n"
            "解决方式：\n"
            " - 把这些文件放到上面的 base_path 目录下；或\n"
            " - 运行时用 --base-path 指向你真实的数据文件夹。"
        )


def _require_any_files(base_path: str, candidates: Iterable[str], *, what: str) -> None:
    """
    至少需要存在 candidates 里的任意一个文件（否则给出缺失提示）。
    用于 trend 这类“多文件可选加载”的场景。
    """
    existing = [f for f in candidates if os.path.exists(os.path.join(base_path, f))]
    if existing:
        return
    missing_list = "\n".join([f" - {m}" for m in candidates])
    raise FileNotFoundError(
        f"{what} 在目录中没有找到任何预期输入文件（当前 data_dir: {base_path}）。至少需要其中之一：\n{missing_list}\n\n"
        "解决方式：\n"
        " - 把数据文件放到上面的 data_dir；或\n"
        " - 运行时用 --data-dir 指向你真实的数据文件夹。"
    )


def _set_cn_font_for_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _safe_read_excel(path: str):
    import pandas as pd

    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _try_write_plotly_image(fig, png_path: str, *, width: int = 1400, height: int = 800, scale: int = 2) -> bool:
    """
    plotly 导出 PNG 通常依赖 kaleido；如果没装，降级成只输出 HTML。
    """
    try:
        fig.write_image(png_path, width=width, height=height, scale=scale)
        return True
    except Exception:
        return False


# =========================
# 10 月脚本整合
# =========================


def viz10(base_path: str) -> None:
    """
    原 `5508 visual code/10/scam_visualization.py` 的整合版：
    - 类型分布堆叠柱状图
    - Top10 地区 vs 类型热力图
    - 金额分布箱线图（log）
    - 全球分布 choropleth（同时输出 html + 尝试 png）
    """
    _require_path_exists(base_path, "10月 BASE_PATH")
    _require_files(
        base_path,
        [
            "1. sms_scam_types.xlsx",
            "2. news_scam_types.xlsx",
            "3. social_media_scam_types.xlsx",
            "1. news_scam_cases_2025_10.xlsx",
        ],
        what="viz10",
    )
    _set_cn_font_for_matplotlib()

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-not-found]
    import plotly.express as px

    print("正在读取数据...")
    sms_type = pd.read_excel(os.path.join(base_path, "1. sms_scam_types.xlsx"))
    news_type = pd.read_excel(os.path.join(base_path, "2. news_scam_types.xlsx"))
    social_type = pd.read_excel(os.path.join(base_path, "3. social_media_scam_types.xlsx"))
    news_case = pd.read_excel(os.path.join(base_path, "1. news_scam_cases_2025_10.xlsx"))

    # 合并新闻案件与类型表
    if "case_id" in news_case.columns and "case_id" in news_type.columns and "primary_type" in news_type.columns:
        news_case = pd.merge(news_case, news_type[["case_id", "primary_type"]], on="case_id", how="left")

    print(f"✅ 读取完成：SMS {len(sms_type)}条 | News {len(news_case)}条 | Social {len(social_type)}条")

    print("生成图1：诈骗类型分布...")
    type_data = {
        "SMS": sms_type["primary_type"].value_counts() if "primary_type" in sms_type.columns else pd.Series(dtype=int),
        "News": news_type["primary_type"].value_counts() if "primary_type" in news_type.columns else pd.Series(dtype=int),
        "Social Media": social_type["primary_type"].value_counts()
        if "primary_type" in social_type.columns
        else pd.Series(dtype=int),
    }
    df_type = pd.DataFrame(type_data).fillna(0).T

    plt.figure(figsize=(12, 6))
    df_type.plot(kind="bar", stacked=True, colormap="tab10")
    plt.title("诈骗类型分布（按数据源）", fontsize=16, pad=20)
    plt.ylabel("案件数量")
    plt.xlabel("数据来源")
    plt.legend(title="Primary Scam Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "viz_1_primary_type_by_source.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("生成图2：新闻诈骗高发地区 vs 类型...")
    if "location" in news_case.columns and "primary_type" in news_case.columns:
        top_locations = news_case["location"].explode().value_counts().head(10).index.tolist()
        news_filtered = news_case[news_case["location"].apply(lambda x: any(loc in str(x) for loc in top_locations))]
        heatmap_data = (
            pd.crosstab(news_filtered["location"].explode(), news_filtered["primary_type"]).loc[top_locations]
            if len(top_locations) > 0
            else None
        )
        if heatmap_data is not None:
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
            plt.title("新闻诈骗高发地区 vs 主要诈骗类型（Top 10地区）", fontsize=16)
            plt.xlabel("Primary Scam Type")
            plt.ylabel("地区 (Top 10)")
            plt.tight_layout()
            plt.savefig(os.path.join(base_path, "viz_2_news_location_vs_type.png"), dpi=300, bbox_inches="tight")
            plt.close()
    else:
        print("⚠️ 跳过图2：news_case 缺少 location/primary_type 列。")

    print("生成图3：涉案金额分布...")
    if "amount" in news_case.columns and "primary_type" in news_case.columns:
        news_case["amount_clean"] = pd.to_numeric(news_case["amount"], errors="coerce")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=news_case, x="primary_type", y="amount_clean", palette="Set2")
        plt.title("新闻诈骗涉案金额分布（按主要类型）", fontsize=16)
        plt.ylabel("金额 (USD)")
        plt.xlabel("Primary Scam Type")
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "viz_3_news_amount_by_type_clean.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("⚠️ 跳过图3：news_case 缺少 amount/primary_type 列。")

    print("生成图4：世界热力图（蓝色渐变）...")
    if "location" not in news_case.columns:
        print("⚠️ 跳过图4：news_case 缺少 location 列。")
        return

    country_map = {
        "Mumbai": "India",
        "New Delhi": "India",
        "Hyderabad": "India",
        "India": "India",
        "United Kingdom": "United Kingdom",
        "UK": "United Kingdom",
        "Australia": "Australia",
        "Cambodia": "Cambodia",
        "Singapore": "Singapore",
        "Thiruvananthapuram": "India",
        "Delhi": "India",
        "Kolkata": "India",
    }

    news_case = news_case.copy()
    news_case["country"] = news_case["location"].explode().map(country_map).fillna("Other")
    country_count = news_case.groupby("country").size().reset_index(name="case_count")

    fig = px.choropleth(
        country_count,
        locations="country",
        locationmode="country names",
        color="case_count",
        hover_name="country",
        color_continuous_scale="Blues",
        title="新闻诈骗案件世界热力图（蓝色越深 = 案件越多）",
    )
    fig.update_layout(
        geo=dict(
            showcoastlines=True,
            coastlinecolor="#ffffff",
            showland=True,
            landcolor="#f8f8f8",
            showocean=True,
            oceancolor="#e6f0ff",
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(title="案件数量", thickness=20, len=0.6),
        title_font_size=18,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html_path = os.path.join(base_path, "viz_4_world_map_news_scams.html")
    png_path = os.path.join(base_path, "viz_4_world_map_news_scams.png")
    fig.write_html(html_path)
    wrote_png = _try_write_plotly_image(fig, png_path)

    print("🎉 全部生成完成！")
    print("输出文件：")
    print(f" - {os.path.join(base_path, 'viz_1_primary_type_by_source.png')}")
    print(f" - {os.path.join(base_path, 'viz_2_news_location_vs_type.png')}")
    print(f" - {os.path.join(base_path, 'viz_3_news_amount_by_type_clean.png')}")
    print(f" - {html_path}")
    if wrote_png:
        print(f" - {png_path}")
    else:
        print(" - ⚠️ 未导出 PNG（通常是缺少 kaleido），HTML 已生成。")


def heatmap10_clean(output_path: Optional[str] = None) -> None:
    """
    原 `5508 visual code/10/worldmap-viz.py` 实际是“干净英文热图”的重建版。
    这个函数不依赖 Excel 文件，可直接生成图片。
    """
    import pandas as pd
    import seaborn as sns  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt

    scam_types = [
        "Advertisement Scam",
        "Financial Scam",
        "Identity-Based Scam",
        "Impersonation Scam",
        "Investment Scam",
        "Market Scam",
        "Other",
        "Relationship-Based Scam",
        "Romance/Emotional Scam",
    ]
    victim_groups = [
        "In-Depth Reports (News)",
        "Mobile Users (SMS)",
        "Social Media (Active Users)",
        "Policy & News Reports",
    ]
    data = [
        [38, 398, 231, 246, 284, 78, 1, 0, 0],
        [105, 206, 0, 0, 583, 245, 1, 138, 0],
        [27, 13, 20, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    df = pd.DataFrame(data, index=victim_groups, columns=scam_types)
    plt.figure(figsize=(16, 9))
    sns.set_style("white")

    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Case Frequency"},
    )
    plt.title("Diffusion Mapping: Scam Tactics vs. Target Demographics", fontsize=18, pad=25, fontweight="bold")
    plt.xlabel("Refined Scam Taxonomy", fontsize=14)
    plt.ylabel("User Group / Data Source", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out = output_path or os.path.join(_script_dir(), "Clean_English_Scam_Heatmap_Final.png")
    plt.savefig(out, dpi=400, bbox_inches="tight")
    print(f"✅ 图片已生成：{out}")


def worldmap10(base_path: str) -> None:
    """
    原 `5508 visual code/10/heatmap-viz.py` 的“全球热力图（优化版）”。
    """
    _require_path_exists(base_path, "10月 BASE_PATH")
    _require_files(
        base_path,
        [
            "2. news_scam_types.xlsx",
            "1. news_scam_cases_2025_10.xlsx",
        ],
        what="worldmap10",
    )
    import ast
    import pandas as pd
    import plotly.express as px

    news_type = pd.read_excel(os.path.join(base_path, "2. news_scam_types.xlsx"))
    news_case = pd.read_excel(os.path.join(base_path, "1. news_scam_cases_2025_10.xlsx"))
    if "case_id" in news_case.columns and "case_id" in news_type.columns and "primary_type" in news_type.columns:
        news_case = pd.merge(news_case, news_type[["case_id", "primary_type"]], on="case_id", how="left")

    def parse_and_map(location_raw: Any) -> Optional[str]:
        mapping = {
            "India": "India",
            "New Delhi": "India",
            "Delhi": "India",
            "Mumbai": "India",
            "Haryana": "India",
            "China": "China",
            "中国": "China",
            "中國": "China",
            "湖北": "China",
            "武汉": "China",
            "江苏": "China",
            "南京": "China",
            "云南": "China",
            "青海": "China",
            "Hong Kong": "China",
            "香港": "China",
            "尖沙咀": "China",
            "Macau": "China",
            "澳門": "China",
            "Taiwan": "Taiwan",
            "台灣": "Taiwan",
            "台湾": "Taiwan",
            "台北": "Taiwan",
            "Taipei": "Taiwan",
            "Singapore": "Singapore",
            "新加坡": "Singapore",
            "Cambodia": "Cambodia",
            "柬埔寨": "Cambodia",
            "Myanmar": "Myanmar",
            "緬甸": "Myanmar",
            "缅北": "Myanmar",
            "Thailand": "Thailand",
            "泰國": "Thailand",
            "泰国": "Thailand",
            "Vietnam": "Vietnam",
            "越南": "Vietnam",
            "Malaysia": "Malaysia",
            "馬來西亞": "Malaysia",
            "Philippines": "Philippines",
            "菲律賓": "Philippines",
            "Palau": "Palau",
            "帕勞": "Palau",
            "Timor-Leste": "Timor-Leste",
            "東帝汶": "Timor-Leste",
            "United States": "United States",
            "USA": "United States",
            "美國": "United States",
            "美国": "United States",
            "Canada": "Canada",
            "加拿大": "Canada",
            "Brazil": "Brazil",
            "巴西": "Brazil",
            "United Kingdom": "United Kingdom",
            "UK": "United Kingdom",
            "英國": "United Kingdom",
            "英国": "United Kingdom",
            "London": "United Kingdom",
            "倫敦": "United Kingdom",
            "Wembley": "United Kingdom",
            "Sweden": "Sweden",
            "瑞典": "Sweden",
            "Germany": "Germany",
            "德國": "Germany",
            "德国": "Germany",
            "卡爾斯魯厄": "Germany",
            "France": "France",
            "法國": "France",
            "Italy": "Italy",
            "意大利": "Italy",
            "UAE": "United Arab Emirates",
            "阿聯酋": "United Arab Emirates",
            "阿联酋": "United Arab Emirates",
            "Dubai": "United Arab Emirates",
            "Nigeria": "Nigeria",
            "尼日利亞": "Nigeria",
            "South Africa": "South Africa",
            "南非": "South Africa",
            "Egypt": "Egypt",
            "埃及": "Egypt",
            "Australia": "Australia",
            "澳大利亞": "Australia",
            "澳大利亚": "Australia",
            "Sydney": "Australia",
        }
        try:
            if isinstance(location_raw, str) and location_raw.startswith("["):
                loc_list = ast.literal_eval(location_raw)
            elif isinstance(location_raw, list):
                loc_list = location_raw
            else:
                loc_list = [str(location_raw)]

            matched = set()
            for item in loc_list:
                item_str = str(item).strip()
                for key, country in mapping.items():
                    if key.lower() in item_str.lower():
                        matched.add(country)
            return list(matched)[0] if matched else None
        except Exception:
            return None

    df_map = news_case.copy()
    if "location" not in df_map.columns:
        raise KeyError("news_case 缺少 location 列，无法生成地图。")
    df_map["matched_country"] = df_map["location"].apply(parse_and_map)
    country_count = (
        df_map.dropna(subset=["matched_country"])
        .groupby("matched_country")
        .size()
        .reset_index(name="case_count")
        .rename(columns={"matched_country": "country"})
    )

    print("\n--- 最终地图数据统计 ---")
    print(country_count.sort_values(by="case_count", ascending=False).to_string(index=False))

    fig = px.choropleth(
        country_count,
        locations="country",
        locationmode="country names",
        color="case_count",
        hover_name="country",
        color_continuous_scale="Blues",
        range_color=[0, float(country_count["case_count"].quantile(0.9))] if len(country_count) else [0, 1],
        title="全球詐騙案件分佈熱力圖（基於新聞案件庫）",
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            landcolor="#f3f3f3",
            oceancolor="#e6f0ff",
            showocean=True,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html_path = os.path.join(base_path, "viz_4_world_map_news_scams.html")
    png_path = os.path.join(base_path, "viz_4_world_map_news_scams.png")
    fig.write_html(html_path)
    wrote_png = _try_write_plotly_image(fig, png_path)

    if wrote_png:
        print(f"✅ PNG图片生成成功：{png_path}")
    else:
        print("⚠️ PNG生成失败（通常是缺少 kaleido），但 HTML 已更新。")
    print(f"🎉 任务完成！请查看：{html_path}")


def lda10(base_path: str, *, n_topics: int = 5) -> None:
    """
    原 `5508 visual code/10/lda_scam_topic_modeling.py.py` 的整合版（修复了 f-string 语法问题）。
    输入：base_path 下的
      - 1. sms_scripts.xlsx
      - 2. news_scripts.xlsx
      - 3. social_media_scripts.xlsx
    """
    _require_path_exists(base_path, "10月 BASE_DIR")
    _require_files(
        base_path,
        [
            "1. sms_scripts.xlsx",
            "2. news_scripts.xlsx",
            "3. social_media_scripts.xlsx",
        ],
        what="lda10",
    )

    import pandas as pd
    import jieba  # type: ignore[import-not-found]
    import re
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    def load_and_merge_data(file_names: Iterable[str]) -> "pd.DataFrame":
        dfs = []
        for f_name in file_names:
            full_path = os.path.join(base_path, f_name)
            if os.path.exists(full_path):
                print(f"正在读取 Excel 文件: {f_name}")
                try:
                    df = pd.read_excel(full_path)
                    if "script_pattern" in df.columns:
                        dfs.append(df[["script_pattern"]].dropna())
                    else:
                        print(f"警告：文件 {f_name} 中没有找到 'script_pattern' 列")
                except Exception as e:
                    print(f"读取 {f_name} 出错: {e}")
            else:
                print(f"找不到文件: {full_path}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def chinese_preprocessing(text: Any) -> str:
        text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
        words = jieba.cut(text)
        return " ".join([w for w in words if len(w) > 1])

    def run_topic_modeling(data: "pd.DataFrame") -> None:
        if data.empty:
            print("没有可分析的数据内容。")
            return

        print("正在进行数据清洗与分词...")
        data = data.copy()
        data["cleaned_text"] = data["script_pattern"].apply(chinese_preprocessing)

        vectorizer = CountVectorizer(max_features=1000)
        tf = vectorizer.fit_transform(data["cleaned_text"])

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_results = lda.fit_transform(tf)

        words = vectorizer.get_feature_names_out()
        topic_keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-11:-1]]
            topic_keywords.append({"主题": f"主题 {topic_idx + 1}", "核心关键词": " / ".join(top_words)})

        data["所属主题"] = lda_results.argmax(axis=1) + 1

        output_path = os.path.join(base_path, "诈骗话术分析结果汇总.xlsx")
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(topic_keywords).to_excel(writer, sheet_name="主题概览", index=False)
            data[["script_pattern", "所属主题"]].sort_values("所属主题").to_excel(
                writer, sheet_name="话术详细分类", index=False
            )

        print("\n" + "=" * 30)
        print(f"✅ 分析完成！结果已保存至:\n{output_path}")
        print("=" * 30)

    files = ["1. sms_scripts.xlsx", "2. news_scripts.xlsx", "3. social_media_scripts.xlsx"]
    df_all = load_and_merge_data(files)
    if not df_all.empty:
        print(f"\n成功汇总所有数据，总计 {len(df_all)} 条话术模式。")
        run_topic_modeling(df_all)
    else:
        print("\n[错误] 未能读取到任何数据。请确认：")
        print(f"1. 文件夹 {base_path} 下是否存在这些文件")
        print("2. Excel 文件内的列名是否确实叫 'script_pattern'")


# =========================
# 11 月脚本整合
# =========================


def viz11(base_path: str) -> None:
    """
    原 `5508 visual code/11/visual11.py` 的整合版：
    - 主要类型分布（Social vs News）堆叠柱状图
    - 金额箱线图（如果 amount 列存在）
    """
    _require_path_exists(base_path, "11月 BASE_PATH")
    _require_files(
        base_path,
        [
            "social_patterns_scam_types_2025_11.xlsx",
            "news_scam_types_2025_11.xlsx",
        ],
        what="viz11",
    )
    _set_cn_font_for_matplotlib()

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-not-found]

    files = {
        "social_scam_types": "social_patterns_scam_types_2025_11.xlsx",
        "news_types": "news_scam_types_2025_11.xlsx",
        "news_tactics": "news_tactic_categories_ai_2025_11.xlsx",  # 可选
        "social_final": "social_scam_patterns_final_2025_11.xlsx",  # 可选
    }

    def load_file(filename: str):
        path = os.path.join(base_path, filename)
        print(f"读取 → {filename}")
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return None
        return pd.read_excel(path)

    print("正在读取文件...\n")
    social_scam = load_file(files["social_scam_types"])
    news_scam = load_file(files["news_types"])

    if social_scam is None or news_scam is None:
        print("❌ 关键文件缺失，无法继续。")
        return

    print("\n✅ 文件加载成功！开始生成图表...\n")

    plt.figure(figsize=(12, 7))
    type_data = pd.DataFrame(
        {
            "Social Media": social_scam["primary_type"].value_counts()
            if "primary_type" in social_scam.columns
            else pd.Series(dtype=int),
            "News Cases": news_scam["primary_type"].value_counts()
            if "primary_type" in news_scam.columns
            else pd.Series(dtype=int),
        }
    ).fillna(0)
    type_data.plot(kind="bar", stacked=True, colormap="tab20")
    plt.title("11月诈骗主要类型分布 (Social vs News)", fontsize=15)
    plt.ylabel("数量")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="数据来源")
    plt.tight_layout()
    out1 = os.path.join(base_path, "viz_1_scam_type_distribution.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()

    out2 = None
    if "amount" in news_scam.columns and "primary_type" in news_scam.columns:
        news_scam = news_scam.copy()
        news_scam["amount_clean"] = pd.to_numeric(news_scam["amount"], errors="coerce")
        plt.figure(figsize=(13, 7))
        sns.boxplot(
            data=news_scam.dropna(subset=["amount_clean"]),
            x="primary_type",
            y="amount_clean",
            palette="Set3",
        )
        plt.yscale("log")
        plt.title("News 诈骗类型涉案金额分布 (对数尺度)", fontsize=15)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out2 = os.path.join(base_path, "viz_2_amount_distribution.png")
        plt.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close()

    print("🎉 图表生成完成！")
    print("输出文件：")
    print(f" - {out1}")
    if out2:
        print(f" - {out2}")
    else:
        print(" - （未生成金额分布：news 文件缺少 amount 列）")


def heatmap11(base_path: str) -> None:
    """
    原 `5508 visual code/11/heatmap11.py`：News vs Social 的类型分布热图（更稳健的 groupby 版本）。
    """
    _require_path_exists(base_path, "11月 BASE_PATH")
    _require_files(
        base_path,
        [
            "news_scam_types_2025_11.xlsx",
            "social_scam_patterns_final_2025_11.xlsx",
        ],
        what="heatmap11",
    )
    _set_cn_font_for_matplotlib()

    import pandas as pd
    import seaborn as sns  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt

    files = {"News": "news_scam_types_2025_11.xlsx", "Social": "social_scam_patterns_final_2025_11.xlsx"}

    all_data = []
    print("🚀 开始读取并清洗数据...")
    for source_label, filename in files.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            df = _safe_read_excel(file_path)
            temp_df = pd.DataFrame()
            if "primary_type" in df.columns:
                temp_df["Scam_Type"] = df["primary_type"]
            elif "scam_type" in df.columns:
                temp_df["Scam_Type"] = df["scam_type"]
            else:
                print(f"⚠️ 跳过：{filename} 缺少 primary_type/scam_type 列")
                continue
            temp_df["Source"] = source_label
            all_data.append(temp_df)
        else:
            print(f"⚠️ 跳过缺失文件: {filename}")

    if not all_data:
        print("❌ 错误：未读取到任何有效数据，请检查路径和文件名。")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined["Scam_Type"] = combined["Scam_Type"].astype(str).str.strip()
    pivot_table = combined.groupby(["Scam_Type", "Source"]).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=True, fmt="g", cmap="YlGnBu", cbar_kws={"label": "案件数量"})
    plt.title("11月 诈骗类型分布热图 (News vs Social)", fontsize=16, pad=20)
    plt.xlabel("数据来源", fontsize=12)
    plt.ylabel("诈骗类型", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(base_path, "Nov_Comparison_Heatmap_Fixed.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 修复后的热图已保存至: {save_path}")


def worldmap11(base_path: str) -> None:
    """
    原 `5508 visual code/11/worldmap11.py`：11月诈骗分布图（中国大陆与港澳台独立显示）。
    """
    _require_path_exists(base_path, "11月 BASE_PATH")
    _require_files(base_path, ["news_scam_types_2025_11.xlsx"], what="worldmap11")
    import pandas as pd
    import plotly.express as px
    import ast

    filename = "news_scam_types_2025_11.xlsx"
    file_path = os.path.join(base_path, filename)

    def parse_and_map_split(location_raw: Any) -> Optional[str]:
        mapping = {
            "China": "China",
            "中国大陆": "China",
            "湖北": "China",
            "武汉": "China",
            "Hong Kong": "Hong Kong",
            "香港": "Hong Kong",
            "HK": "Hong Kong",
            "Macau": "Macao",
            "澳门": "Macao",
            "澳門": "Macao",
            "Taiwan": "Taiwan",
            "台湾": "Taiwan",
            "台灣": "Taiwan",
            "Singapore": "Singapore",
            "India": "India",
            "USA": "United States",
            "United Kingdom": "United Kingdom",
            "Cambodia": "Cambodia",
            "Myanmar": "Myanmar",
        }
        try:
            if pd.isna(location_raw):
                return None
            if isinstance(location_raw, str) and location_raw.startswith("["):
                loc_list = ast.literal_eval(location_raw)
            else:
                loc_list = [str(location_raw)]
            for item in loc_list:
                item_str = str(item).strip()
                for key, country in mapping.items():
                    if key.lower() in item_str.lower():
                        return country
            return None
        except Exception:
            return None

    print(f"🚀 正在分析路径: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 未找到文件，请检查：{file_path}")

    df = _safe_read_excel(file_path)
    if "location" not in df.columns:
        raise KeyError(f"{filename} 缺少 location 列，无法生成地图。")

    df = df.copy()
    df["matched_region"] = df["location"].apply(parse_and_map_split)
    region_count = df.dropna(subset=["matched_region"]).groupby("matched_region").size().reset_index(name="count")
    print("📊 统计结果预览（已分开）：")
    print(region_count.to_string(index=False))

    fig = px.choropleth(
        region_count,
        locations="matched_region",
        locationmode="country names",
        color="count",
        hover_name="matched_region",
        color_continuous_scale="Reds",
        title="11月诈骗分布图 (中国大陆与港澳台独立显示)",
    )
    output_map = os.path.join(base_path, "viz_map_split_nov.html")
    fig.write_html(output_map)
    print(f"✅ 交互地图已保存: {output_map}")


def lda11(base_path: str, *, n_topics: int = 5) -> None:
    """
    原 `5508 visual code/11/scam_topic_model11.py`：自动扫描文件夹下所有 xlsx，提取 script_pattern 做 LDA。
    """
    _require_path_exists(base_path, "11月 BASE_DIR")

    import pandas as pd
    import jieba  # type: ignore[import-not-found]
    import re
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    output_file = os.path.join(base_path, "11月诈骗话术深度分析报告.xlsx")

    stop_words = {
        "通过",
        "利用",
        "进行",
        "虚假",
        "实施",
        "要求",
        "以及",
        "可以",
        "或者",
        "由于",
        "进入",
        "引导",
        "点击",
        "链接",
        "网站",
        "平台",
        "相关",
        "提供",
        "所谓",
        "对方",
        "用户",
        "受害者",
        "已经",
        "目前",
        "发现",
    }

    def load_all_xlsx(folder_path: str) -> "pd.DataFrame":
        dfs = []
        files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
        for f_name in files:
            full_path = os.path.join(folder_path, f_name)
            try:
                df = pd.read_excel(full_path)
                if "script_pattern" in df.columns:
                    print(f"成功读取: {f_name}")
                    dfs.append(df[["script_pattern"]].dropna())
                else:
                    print(f"跳过: {f_name} (未找到 script_pattern 列)")
            except Exception as e:
                print(f"读取 {f_name} 出错: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def chinese_preprocessing(text: Any) -> str:
        text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
        words = jieba.cut(text)
        return " ".join([w for w in words if len(w) > 1 and w not in stop_words])

    def run_topic_modeling(data: "pd.DataFrame") -> None:
        print("\n正在处理文本数据...")
        data = data.copy()
        data["cleaned_text"] = data["script_pattern"].apply(chinese_preprocessing)
        vectorizer = CountVectorizer(max_features=1000)
        tf = vectorizer.fit_transform(data["cleaned_text"])
        print(f"开始 LDA 主题建模 (目标类别: {n_topics})...")
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_results = lda.fit_transform(tf)

        words = vectorizer.get_feature_names_out()
        topic_summary = []
        for idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-11:-1]]
            topic_summary.append({"主题编号": f"主题 {idx + 1}", "核心关键词": " / ".join(top_words)})

        data["所属主题编号"] = lda_results.argmax(axis=1) + 1
        print("正在生成分析报告...")
        with pd.ExcelWriter(output_file) as writer:
            pd.DataFrame(topic_summary).to_excel(writer, sheet_name="主题特征定义", index=False)
            data[["script_pattern", "所属主题编号"]].to_excel(writer, sheet_name="话术自动分类明细", index=False)

        print("\n" + "=" * 40)
        print(f"✅ 分析完成！报告已保存至:\n{output_file}")
        print("=" * 40 + "\n主题预览:")
        for t in topic_summary:
            print(f"{t['主题编号']}: {t['核心关键词']}")

    df_all = load_all_xlsx(base_path)
    if not df_all.empty:
        run_topic_modeling(df_all)
    else:
        print("文件夹内未找到有效数据，请检查路径。")


# =========================
# 12 月脚本整合
# =========================


def viz12(base_path: str) -> None:
    """
    原 `5508 visual code/12/visual12.py`：12月类型热图 + 柱状图 + 金额分布。
    """
    _require_path_exists(base_path, "12月 BASE_PATH")
    _require_files(
        base_path,
        [
            "news_scam_types.xlsx",
            "social_patterns_scam_types.xlsx",
        ],
        what="viz12",
    )
    _set_cn_font_for_matplotlib()

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-not-found]

    files = {
        "news": "news_scam_types.xlsx",
        "social": "social_patterns_scam_types.xlsx",
    }

    def load_file(filename: str):
        path = os.path.join(base_path, filename)
        print(f"尝试读取 → {filename}")
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return None
        try:
            df = pd.read_excel(path)
            print(f"✅ 成功加载: {filename}  ({len(df)} 行)")
            return df
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            return None

    print("🚀 开始读取 12月诈骗数据...\n")
    news_df = load_file(files["news"])
    social_df = load_file(files["social"])

    if news_df is None or social_df is None:
        print("❌ 关键文件加载失败，请检查文件夹和文件名。")
        return

    print("\n✅ 主要数据加载成功！开始生成图表...\n")
    combined = []
    if "primary_type" in news_df.columns:
        combined.append(pd.DataFrame({"Scam_Type": news_df["primary_type"], "Source": "News"}))
    if "primary_type" in social_df.columns:
        combined.append(pd.DataFrame({"Scam_Type": social_df["primary_type"], "Source": "Social"}))
    if not combined:
        print("❌ News/Social 都没有 primary_type 列，无法绘图。")
        return

    combined_df = pd.concat(combined, ignore_index=True)
    combined_df["Scam_Type"] = combined_df["Scam_Type"].astype(str).str.strip()
    pivot = combined_df.groupby(["Scam_Type", "Source"]).size().unstack(fill_value=0)

    import matplotlib.pyplot as plt  # noqa: F811

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt="g", cmap="YlGnBu", cbar_kws={"label": "案例数量"})
    plt.title("12月 诈骗类型分布热图 (News vs Social)", fontsize=16)
    plt.xlabel("数据来源")
    plt.ylabel("诈骗类型")
    plt.tight_layout()
    out1 = os.path.join(base_path, "12月_诈骗类型热图.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    type_data = pd.DataFrame(
        {
            "News": news_df["primary_type"].value_counts() if "primary_type" in news_df.columns else pd.Series(dtype=int),
            "Social": social_df["primary_type"].value_counts()
            if "primary_type" in social_df.columns
            else pd.Series(dtype=int),
        }
    ).fillna(0)
    type_data.plot(kind="bar", stacked=True, colormap="tab20")
    plt.title("12月诈骗主要类型分布 (News vs Social)", fontsize=15)
    plt.ylabel("数量")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="数据来源")
    plt.tight_layout()
    out2 = os.path.join(base_path, "12月_类型柱状图.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()

    out3 = None
    if "amount" in news_df.columns and "primary_type" in news_df.columns:
        news_df = news_df.copy()
        news_df["amount_clean"] = pd.to_numeric(news_df["amount"], errors="coerce")
        plt.figure(figsize=(13, 7))
        sns.boxplot(data=news_df.dropna(subset=["amount_clean"]), x="primary_type", y="amount_clean", palette="Set3")
        plt.yscale("log")
        plt.title("12月 News 诈骗类型涉案金额分布 (对数尺度)", fontsize=15)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out3 = os.path.join(base_path, "12月_金额分布.png")
        plt.savefig(out3, dpi=300, bbox_inches="tight")
        plt.close()

    print("\n🎉 图表生成完成！输出文件：")
    print(f" - {out1}")
    print(f" - {out2}")
    if out3:
        print(f" - {out3}")
    else:
        print(" - （未生成金额分布：news 文件缺少 amount 列）")


def worldmap12(base_path: str) -> None:
    """
    原 `5508 visual code/12/worldmap12.py`：12月全球诈骗分布地图（尽量使用真实 location/country 列）。
    """
    _require_path_exists(base_path, "12月 BASE_PATH")
    _require_files(base_path, ["news_scam_types.xlsx"], what="worldmap12")
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    filename = "news_scam_types.xlsx"
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件：{path}")
    news_df = pd.read_excel(path)

    print("\n🔍 正在统计各国家/地区诈骗数量...")
    if "location" in news_df.columns:
        location_col = "location"
    elif "country" in news_df.columns:
        location_col = "country"
    else:
        location_col = None
        print("⚠️ news_scam_types.xlsx 中没有找到 'location' 或 'country' 列。")

    if location_col:
        region_counts = news_df[location_col].value_counts().reset_index()
        region_counts.columns = ["region", "count"]
        print(f"\n✅ 共统计到 {len(region_counts)} 个不同地区，前15个：")
        print(region_counts.head(15).to_string(index=False))
    else:
        region_counts = pd.DataFrame({"region": [], "count": []})

    if len(region_counts) == 0:
        print("⚠️ 使用手动示例数据（请尽快补充 location 列）")
        region_counts = pd.DataFrame(
            {
                "region": [
                    "China",
                    "Hong Kong",
                    "Macao",
                    "Taiwan",
                    "Cambodia",
                    "Myanmar",
                    "Singapore",
                    "India",
                    "United Kingdom",
                    "United States",
                ],
                "count": [180, 52, 15, 42, 9, 18, 25, 12, 7, 35],
            }
        )

    fig_map = px.choropleth(
        region_counts,
        locations="region",
        locationmode="country names",
        color="count",
        hover_name="region",
        color_continuous_scale="Reds",
        title="12月全球诈骗分布图（真实数据 · 全球视角）",
        labels={"count": "诈骗案例数量"},
    )
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        margin={"r": 0, "t": 70, "l": 0, "b": 0},
        height=720,
        title_font_size=20,
    )

    map_path = os.path.join(base_path, "12月_全球诈骗分布地图.html")
    pio.write_html(fig_map, file=map_path, auto_open=False, include_plotlyjs="cdn")
    print(f"\n✅ 全球地图已生成：{map_path}")


def lda12(base_path: str, *, month: str = "12", n_topics: int = 5) -> None:
    """
    原 `5508 visual code/12/lda_scam_topic_modeling12.py`：自动扫描 xlsx，提取 script_pattern，输出 LDA 报告。
    """
    _require_path_exists(base_path, "12月 BASE_DIR")

    import pandas as pd
    import jieba  # type: ignore[import-not-found]
    import re
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    output_file = os.path.join(base_path, f"{month}月诈骗话术深度分析报告.xlsx")
    stop_words = {
        "通过",
        "利用",
        "进行",
        "虚假",
        "实施",
        "要求",
        "以及",
        "可以",
        "或者",
        "由于",
        "进入",
        "引导",
        "点击",
        "链接",
        "网站",
        "平台",
        "相关",
        "提供",
        "所谓",
        "对方",
        "用户",
        "受害者",
        "发现",
        "已经",
        "目前",
        "一种",
        "背后",
    }

    def load_all_xlsx(folder_path: str) -> "pd.DataFrame":
        dfs = []
        files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
        for f_name in files:
            full_path = os.path.join(folder_path, f_name)
            try:
                df = pd.read_excel(full_path)
                if "script_pattern" in df.columns:
                    print(f"成功读取: {f_name}")
                    dfs.append(df[["script_pattern"]].dropna())
                else:
                    print(f"跳过: {f_name} (列名不匹配)")
            except Exception as e:
                print(f"读取 {f_name} 出错: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def chinese_preprocessing(text: Any) -> str:
        text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
        words = jieba.cut(text)
        return " ".join([w for w in words if len(w) > 1 and w not in stop_words])

    def run_topic_modeling(data: "pd.DataFrame") -> None:
        print(f"\n正在对 {month} 月共 {len(data)} 条话术进行预处理...")
        data = data.copy()
        data["cleaned_text"] = data["script_pattern"].apply(chinese_preprocessing)
        vectorizer = CountVectorizer(max_features=1000)
        tf = vectorizer.fit_transform(data["cleaned_text"])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_results = lda.fit_transform(tf)

        words = vectorizer.get_feature_names_out()
        topic_summary = []
        for idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-11:-1]]
            topic_summary.append({"主题编号": f"主题 {idx + 1}", "核心关键词": " / ".join(top_words)})

        data["预测主题编号"] = lda_results.argmax(axis=1) + 1
        with pd.ExcelWriter(output_file) as writer:
            pd.DataFrame(topic_summary).to_excel(writer, sheet_name="主题特征定义", index=False)
            data[["script_pattern", "预测主题编号"]].to_excel(writer, sheet_name="话术分类明细", index=False)

        print("\n" + "=" * 40)
        print(f"✅ {month}月报告生成成功！")
        print(f"文件位置: {output_file}")
        print("=" * 40)
        for t in topic_summary:
            print(f"{t['主题编号']}: {t['核心关键词']}")

    df_all = load_all_xlsx(base_path)
    if not df_all.empty:
        run_topic_modeling(df_all, )
    else:
        print("未发现有效数据。")


# =========================
# 趋势预测（trend.py 整合）
# =========================


def trend(data_dir: str, *, output_dir: Optional[str] = None) -> None:
    """
    整合自 `trend.py`：诈骗趋势预测（最后7天测试集 + 未来7天预测）。

    输出（保存到 output_dir，默认当前目录）：
      - 1_total_comparison_smooth.png
      - 2_type_forecast.png
      - 3_tactic_forecast.png
      - 4_region_forecast.png
      - 5_platform_forecast.png
      - prediction_results.txt
    """
    output_dir = output_dir or _script_dir()
    _require_path_exists(data_dir, "trend data_dir")
    _require_path_exists(output_dir, "trend output_dir")

    # step1: 导入依赖（按需导入，避免没装深度学习库时影响其他命令）
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        import ast
        from collections import Counter

        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"trend 功能缺少依赖：{e}\n\n"
            "建议安装：statsmodels scikit-learn keras tensorflow matplotlib pandas numpy\n"
            "例如：python -m pip install statsmodels scikit-learn keras tensorflow"
        ) from e

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # step2: 定义输入文件（保持与 trend.py 一致）
    news_files = ["news_2025_10_extracted.xlsx", "news_2025_11_extracted.xlsx", "news_2025_12_extracted.xlsx"]
    social_files = [
        "social_media_patterns_2025_10.xlsx",
        "social_media_patterns_2025_11.xlsx",
        "social_media_patterns_2025_12.xlsx",
    ]
    _require_any_files(data_dir, [*news_files, *social_files], what="trend")

    # step3: 加载并清洗新闻/社媒数据，统一字段
    def load_news(filename: str) -> "pd.DataFrame":
        path = os.path.join(data_dir, filename)
        try:
            df = pd.read_excel(path)
            if "is_scam" in df.columns:
                df = df[df["is_scam"] is True] if isinstance(df["is_scam"].iloc[0], bool) else df[df["is_scam"] == True]
            if "publish_time" in df.columns:
                df["date"] = pd.to_datetime(df["publish_time"])
            else:
                return pd.DataFrame()
            if "scam_type" not in df.columns:
                df["scam_type"] = "未知"
            if "tactic_tags" not in df.columns:
                if "tactics_tags" in df.columns:
                    df["tactic_tags"] = df["tactics_tags"]
                else:
                    df["tactic_tags"] = "[]"
            for col, default in [
                ("location", "[]"),
                ("country", "[]"),
                ("platform", "未知"),
                ("amount", np.nan),
                ("time", np.nan),
            ]:
                if col not in df.columns:
                    df[col] = default
            return df[["date", "scam_type", "tactic_tags", "location", "country", "platform", "amount", "time"]]
        except Exception as e:
            print(f"加载新闻失败 {filename}: {e}")
            return pd.DataFrame()

    def load_social(filename: str) -> "pd.DataFrame":
        path = os.path.join(data_dir, filename)
        try:
            df = pd.read_excel(path)
            if "is_scam_related" in df.columns:
                df = df[df["is_scam_related"] == True]
            if "post_time" in df.columns:
                df["date"] = pd.to_datetime(df["post_time"])
            else:
                return pd.DataFrame()
            if "scam_type" not in df.columns:
                df["scam_type"] = "未知"
            if "tactic_tags" not in df.columns:
                df["tactic_tags"] = "[]"
            for col, default in [
                ("location", "[]"),
                ("country", "[]"),
                ("platform", "未知"),
                ("amount", np.nan),
                ("time", np.nan),
                ("engagement", np.nan),
            ]:
                if col not in df.columns:
                    df[col] = default
            return df[
                ["date", "scam_type", "tactic_tags", "location", "country", "platform", "amount", "time", "engagement"]
            ]
        except Exception as e:
            print(f"加载社交媒体失败 {filename}: {e}")
            return pd.DataFrame()

    all_data = []
    for f in news_files:
        df = load_news(f)
        if not df.empty:
            all_data.append(df)
    for f in social_files:
        df = load_social(f)
        if not df.empty:
            all_data.append(df)
    if not all_data:
        raise ValueError("trend：没有成功加载到任何数据（请检查 data_dir 与文件名）。")

    df_all = pd.concat(all_data, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all = df_all.dropna(subset=["date"])
    df_all["date"] = df_all["date"].dt.date
    print(f"总记录数: {len(df_all)}")
    print(f"日期范围: {df_all['date'].min()} 到 {df_all['date'].max()}")

    # step4: 构建 2025-10-01 ~ 2025-12-31 的每日序列（缺天补 0）
    from datetime import datetime, timedelta  # noqa: F811

    start_date = datetime(2025, 10, 1).date()
    end_date = datetime(2025, 12, 31).date()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D").date

    daily_total = df_all.groupby("date").size().reindex(date_range, fill_value=0)
    daily_total.index = pd.to_datetime(daily_total.index)

    # 类型（出现>=10次，至少取3类；否则取前5）
    type_counts = df_all["scam_type"].value_counts()
    main_types = type_counts[type_counts >= 10].index.tolist()
    if len(main_types) < 3:
        main_types = type_counts.head(5).index.tolist()
    daily_types = {}
    for t in main_types:
        s = df_all[df_all["scam_type"] == t].groupby("date").size().reindex(date_range, fill_value=0)
        s.index = pd.to_datetime(s.index)
        daily_types[t] = s

    # 手法 tags（取出现最多的 8 个，优先 cnt>=10）
    all_tags = []
    for tags in df_all["tactic_tags"].dropna():
        if isinstance(tags, list):
            all_tags.extend(tags)
        elif isinstance(tags, str):
            try:
                lst = ast.literal_eval(tags)
                all_tags.extend(lst)
            except Exception:
                parts = tags.replace("[", "").replace("]", "").replace("'", "").split(",")
                all_tags.extend([p.strip() for p in parts if p.strip()])
    tag_counts = Counter(all_tags)
    selected_tags = [tag for tag, cnt in tag_counts.most_common(8) if cnt >= 10]
    if not selected_tags:
        selected_tags = [tag for tag, _ in tag_counts.most_common(8)]
    daily_tags = {}
    for tag in selected_tags:
        mask = df_all["tactic_tags"].apply(lambda x: tag in x if isinstance(x, list) else (tag in str(x)))
        s = df_all[mask].groupby("date").size().reindex(date_range, fill_value=0)
        s.index = pd.to_datetime(s.index)
        daily_tags[tag] = s

    # 地区（优先 country，其次 location；取出现>=5 的前 8）
    def extract_region(row):
        if pd.notna(row.get("country")) and row["country"]:
            if isinstance(row["country"], list) and len(row["country"]) > 0:
                return row["country"][0]
            if isinstance(row["country"], str):
                try:
                    lst = ast.literal_eval(row["country"])
                    if lst:
                        return lst[0]
                except Exception:
                    return row["country"]
        if pd.notna(row.get("location")) and row["location"]:
            if isinstance(row["location"], list) and len(row["location"]) > 0:
                return row["location"][0]
            if isinstance(row["location"], str):
                try:
                    lst = ast.literal_eval(row["location"])
                    if lst:
                        return lst[0]
                except Exception:
                    return row["location"]
        return "未知"

    df_all = df_all.copy()
    df_all["region"] = df_all.apply(extract_region, axis=1)
    df_all = df_all[df_all["region"] != "未知"]
    region_counts = df_all["region"].value_counts()
    main_regions = region_counts[region_counts >= 5].index.tolist()[:8]
    daily_region = {}
    for r in main_regions:
        s = df_all[df_all["region"] == r].groupby("date").size().reindex(date_range, fill_value=0)
        s.index = pd.to_datetime(s.index)
        daily_region[r] = s

    # 平台（过滤 []/未知；取出现>=5 的前 8）
    df_all = df_all[~df_all["platform"].isin(["[]", "未知"])]
    platform_counts = df_all["platform"].value_counts()
    main_platforms = platform_counts[platform_counts >= 5].index.tolist()[:8]
    daily_platform = {}
    for p in main_platforms:
        s = df_all[df_all["platform"] == p].groupby("date").size().reindex(date_range, fill_value=0)
        s.index = pd.to_datetime(s.index)
        daily_platform[p] = s

    # step5: 定义预测函数（LSTM 未来预测 + 方向准确率）
    forecast_steps = 7
    last_date = daily_total.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq="D")

    def lstm_forecast_future(series, steps=7, window=5, epochs=30):
        values = series.values.reshape(-1, 1)
        if len(values) < window + 1 or np.all(values == 0):
            return np.full(steps, 0.0)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
        X, y = [], []
        for i in range(len(scaled) - window):
            X.append(scaled[i : i + window])
            y.append(scaled[i + window])
        X = np.array(X).reshape(-1, window, 1)
        y = np.array(y)
        if len(X) == 0:
            return np.full(steps, np.nan)
        model = Sequential()
        model.add(LSTM(20, input_shape=(window, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=4, verbose=0, callbacks=[early_stop])
        last_seq = scaled[-window:].reshape(1, window, 1)
        pred_scaled = []
        for _ in range(steps):
            nxt = model.predict(last_seq, verbose=0)[0, 0]
            pred_scaled.append(nxt)
            nxt_reshaped = np.array([[nxt]]).reshape(1, 1, 1)
            last_seq = np.append(last_seq[:, 1:, :], nxt_reshaped, axis=1)
        return scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()

    def direction_accuracy(actual, pred):
        if len(actual) < 2 or len(pred) < 2:
            return 0.0
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(pred))
        return float(np.mean(actual_dir == pred_dir))

    # step6: 图1（总量对比：ARIMA vs LSTM，基于7日平滑，测试集最后7天）
    print("\n生成图1：总讨论量对比（基于7日移动平均，测试集最后7天）")
    daily_smooth = daily_total.rolling(window=7, min_periods=1).mean().fillna(daily_total)

    def arima_test_smooth(series, test_size=7):
        train = series[:-test_size]
        test = series[-test_size:]
        try:
            from pmdarima import auto_arima  # optional

            auto_model = auto_arima(
                train, seasonal=False, stepwise=True, trace=False, start_p=0, max_p=3, start_q=0, max_q=3, d=1, max_d=2
            )
            order = auto_model.order
        except Exception:
            order = (1, 1, 1)
        history = list(train)
        test_pred = []
        for t in range(len(test)):
            model = ARIMA(history, order=order)
            fit = model.fit()
            yhat = fit.forecast()[0]
            test_pred.append(yhat)
            history.append(test.iloc[t])
        return np.array(test_pred), test.values

    def lstm_test_smooth(series, test_size=7, window=7, epochs=10):
        train = series[:-test_size]
        test = series[-test_size:]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))
        train_scaled = scaled[: len(train)]
        X_train, y_train = [], []
        for i in range(len(train_scaled) - window):
            X_train.append(train_scaled[i : i + window])
            y_train.append(train_scaled[i + window])
        X_train = np.array(X_train).reshape(-1, window, 1)
        y_train = np.array(y_train)
        if len(X_train) == 0:
            return np.full(test_size, np.nan), test.values
        model = Sequential()
        model.add(LSTM(20, input_shape=(window, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=4, verbose=0, callbacks=[early_stop])
        test_pred = []
        current_seq = train_scaled[-window:].flatten().tolist()
        for i in range(len(test)):
            input_arr = np.array(current_seq).reshape(1, window, 1)
            pred_scaled = model.predict(input_arr, verbose=0)[0, 0]
            test_pred.append(pred_scaled)
            next_real = scaled[len(train) + i, 0]
            current_seq = current_seq[1:] + [next_real]
        test_pred_inv = scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).flatten()
        return test_pred_inv, test.values

    arima_pred, actual_smooth = arima_test_smooth(daily_smooth, test_size=7)
    lstm_pred, _ = lstm_test_smooth(daily_smooth, test_size=7, window=5, epochs=30)
    test_dates = daily_smooth.index[-7:]

    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, actual_smooth, "k-o", label="Actual (7d MA)", linewidth=2)
    plt.plot(test_dates, arima_pred, "r--s", label="ARIMA")
    plt.plot(test_dates, lstm_pred, "g--^", label="LSTM")
    plt.title("Total Scam Volume Forecast Comparison (7-day MA)")
    plt.xlabel("Date")
    plt.ylabel("Smoothed Volume")
    plt.legend()
    plt.grid(True)
    out1 = os.path.join(output_dir, "1_total_comparison_smooth.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    rmse_arima = sqrt(mean_squared_error(actual_smooth, arima_pred))
    rmse_lstm = sqrt(mean_squared_error(actual_smooth, lstm_pred))
    dir_acc_arima = direction_accuracy(actual_smooth, arima_pred)
    dir_acc_lstm = direction_accuracy(actual_smooth, lstm_pred)

    # step7: 四个维度未来7天 LSTM 预测（类型/手法/地区/平台），并画图
    type_futures = {}
    plt.figure(figsize=(14, 6))
    for t in main_types[:5]:
        pred = lstm_forecast_future(daily_types[t], steps=forecast_steps, window=5, epochs=30)
        type_futures[t] = pred
        plt.plot(forecast_index, pred, marker="o", label=str(t))
    plt.title("Next-week Forecast by Scam Type (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Forecast Volume")
    plt.legend()
    plt.grid(True)
    out2 = os.path.join(output_dir, "2_type_forecast.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    tactic_futures = {}
    plt.figure(figsize=(14, 6))
    for tag in selected_tags[:8]:
        pred = lstm_forecast_future(daily_tags[tag], steps=forecast_steps, window=5, epochs=30)
        tactic_futures[tag] = pred
        plt.plot(forecast_index, pred, marker="s", label=str(tag))
    plt.title("Next-week Forecast by Tactic Tag (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Forecast Volume")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    out3 = os.path.join(output_dir, "3_tactic_forecast.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()

    region_futures = {}
    plt.figure(figsize=(14, 6))
    for r in main_regions[:8]:
        pred = lstm_forecast_future(daily_region[r], steps=forecast_steps, window=5, epochs=30)
        region_futures[r] = pred
        plt.plot(forecast_index, pred, marker="^", label=str(r))
    plt.title("Next-week Forecast by Region (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Forecast Volume")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    out4 = os.path.join(output_dir, "4_region_forecast.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()

    platform_futures = {}
    plt.figure(figsize=(14, 6))
    for p in main_platforms[:8]:
        pred = lstm_forecast_future(daily_platform[p], steps=forecast_steps, window=5, epochs=30)
        platform_futures[p] = pred
        plt.plot(forecast_index, pred, marker="d", label=str(p))
    plt.title("Next-week Forecast by Platform (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Forecast Volume")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    out5 = os.path.join(output_dir, "5_platform_forecast.png")
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close()

    # step8: 输出汇总报告 prediction_results.txt
    report_path = os.path.join(output_dir, "prediction_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("诈骗趋势预测结果报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("【模型对比（最后7天测试集，基于7日移动平均）】\n")
        f.write(f"ARIMA - RMSE: {rmse_arima:.2f}, 方向准确率: {dir_acc_arima*100:.1f}%\n")
        f.write(f"LSTM  - RMSE: {rmse_lstm:.2f}, 方向准确率: {dir_acc_lstm*100:.1f}%\n")
        f.write("注：由于数据量有限，绝对误差可能偏大；方向准确率更具参考价值。\n\n")

        lstm_future_total = lstm_forecast_future(daily_total, steps=forecast_steps, window=7, epochs=30)
        f.write("未来7天 LSTM 预测值（总讨论量）:\n")
        for date, val in zip(forecast_index, lstm_future_total):
            f.write(f"  {date.date()}: {val:.1f}\n")
        f.write("\n")

        f.write("【诈骗类型热度预测 (LSTM)】\n")
        for t, pred in type_futures.items():
            past_mean = float(daily_types[t][-7:].mean())
            future_mean = float(np.mean(pred))
            growth = (future_mean - past_mean) / past_mean if past_mean > 0 else 0.0
            f.write(f"{t}: 过去7日均值={past_mean:.2f}, 未来7日均值={future_mean:.2f}, 增长率={growth*100:.1f}%\n")
            f.write(f"  每日预测: {', '.join([f'{v:.1f}' for v in pred])}\n")
        f.write("\n")

        f.write("【诈骗手法趋势预测 (LSTM)】\n")
        for tag, pred in tactic_futures.items():
            past_mean = float(daily_tags[tag][-7:].mean())
            future_mean = float(np.mean(pred))
            growth = (future_mean - past_mean) / past_mean if past_mean > 0 else 0.0
            f.write(f"{tag}: 过去7日均值={past_mean:.2f}, 未来7日均值={future_mean:.2f}, 增长率={growth*100:.1f}%\n")
            f.write(f"  每日预测: {', '.join([f'{v:.1f}' for v in pred])}\n")
        f.write("\n")

        f.write("【地区风险预测 (LSTM)】\n")
        for r, pred in region_futures.items():
            past_mean = float(daily_region[r][-7:].mean())
            future_mean = float(np.mean(pred))
            growth = (future_mean - past_mean) / past_mean if past_mean > 0 else 0.0
            f.write(f"{r}: 过去7日均值={past_mean:.2f}, 未来7日均值={future_mean:.2f}, 增长率={growth*100:.1f}%\n")
            f.write(f"  每日预测: {', '.join([f'{v:.1f}' for v in pred])}\n")
        f.write("\n")

        f.write("【平台传播预测 (LSTM)】\n")
        for p, pred in platform_futures.items():
            past_mean = float(daily_platform[p][-7:].mean())
            future_mean = float(np.mean(pred))
            growth = (future_mean - past_mean) / past_mean if past_mean > 0 else 0.0
            f.write(f"{p}: 过去7日均值={past_mean:.2f}, 未来7日均值={future_mean:.2f}, 增长率={growth*100:.1f}%\n")
            f.write(f"  每日预测: {', '.join([f'{v:.1f}' for v in pred])}\n")

    print("\ntrend：输出完成")
    print("生成文件：")
    for p in [out1, out2, out3, out4, out5, report_path]:
        print(f" - {p}")


# =========================
# CLI
# =========================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="代码整合版.py",
        description="把本目录下分散的可视化/LDA脚本整合为单文件命令行工具。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("viz10", help="10月整套可视化（4张图）")
    p.add_argument("--base-path", default=_default_base_path_for("10"))
    p.set_defaults(_run=lambda args: viz10(args.base_path))

    p = sub.add_parser("worldmap10", help="10月全球分布图（优化 mapping 版）")
    p.add_argument("--base-path", default=_default_base_path_for("10"))
    p.set_defaults(_run=lambda args: worldmap10(args.base_path))

    p = sub.add_parser("heatmap10-clean", help="无需数据文件的干净英文热图")
    p.add_argument("--output", default=None, help="输出 PNG 路径（默认当前目录）")
    p.set_defaults(_run=lambda args: heatmap10_clean(args.output))

    p = sub.add_parser("lda10", help="10月话术 LDA 主题建模")
    p.add_argument("--base-path", default=_default_base_path_for("10"))
    p.add_argument("--topics", type=int, default=5)
    p.set_defaults(_run=lambda args: lda10(args.base_path, n_topics=args.topics))

    p = sub.add_parser("viz11", help="11月类型分布+金额分布")
    p.add_argument("--base-path", default=_default_base_path_for("11"))
    p.set_defaults(_run=lambda args: viz11(args.base_path))

    p = sub.add_parser("heatmap11", help="11月 News vs Social 类型热图")
    p.add_argument("--base-path", default=_default_base_path_for("11"))
    p.set_defaults(_run=lambda args: heatmap11(args.base_path))

    p = sub.add_parser("worldmap11", help="11月地图（大陆/港澳台拆分）")
    p.add_argument("--base-path", default=_default_base_path_for("11"))
    p.set_defaults(_run=lambda args: worldmap11(args.base_path))

    p = sub.add_parser("lda11", help="11月话术 LDA（自动扫描 xlsx）")
    p.add_argument("--base-path", default=_default_base_path_for("11"))
    p.add_argument("--topics", type=int, default=5)
    p.set_defaults(_run=lambda args: lda11(args.base_path, n_topics=args.topics))

    p = sub.add_parser("viz12", help="12月类型热图+柱状图+金额分布")
    p.add_argument("--base-path", default=_default_base_path_for("12"))
    p.set_defaults(_run=lambda args: viz12(args.base_path))

    p = sub.add_parser("worldmap12", help="12月全球分布地图")
    p.add_argument("--base-path", default=_default_base_path_for("12"))
    p.set_defaults(_run=lambda args: worldmap12(args.base_path))

    p = sub.add_parser("lda12", help="12月话术 LDA（自动扫描 xlsx）")
    p.add_argument("--base-path", default=_default_base_path_for("12"))
    p.add_argument("--month", default="12")
    p.add_argument("--topics", type=int, default=5)
    p.set_defaults(_run=lambda args: lda12(args.base_path, month=args.month, n_topics=args.topics))

    p = sub.add_parser("trend", help="诈骗趋势预测（ARIMA/LSTM，最后7天测试 + 未来7天）")
    p.add_argument("--data-dir", default=_auto_detect_trend_data_dir(), help="trend 输入数据目录（包含 news_2025_10_extracted 等）")
    p.add_argument("--output-dir", default=_script_dir(), help="输出目录（默认当前目录）")
    p.set_defaults(_run=lambda args: trend(args.data_dir, output_dir=args.output_dir))

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args._run(args)
        return 0
    except KeyboardInterrupt:
        print("\n⛔ 已取消。")
        return 130
    except Exception as e:
        print(f"\n❌ 运行失败：{e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

