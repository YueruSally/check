# 最终实验结果速查（完整结构路径与精确端点版）

本文件记录论文最终结果口径。英文正文为 `paper/07_results.md`，中文对应稿为 `paper/07_results_CN.md`。历史NSGA-II档案只保留为探索性记录，不再作为最终前沿或门户激活结论的依据。

## 1. 模型与求解口径

- 使用结构限制下枚举得到的全部简单路径：最多八个弧段、最多四次模式转换、不重复节点。
- 每个起点—目的地—门户组合有八条结构路径。
- 三个目的地分别求解，不是共享容量的联合多目的地模型。
- 每个条件求两个精确词典序端点：最低成本、再最短时间；最短时间、再最低成本。
- NY–NJ份额上下界在端点等价区域内单独求解。
- 集中单波次是集中到港参照情景；多波次结果用于检验需求时间分布敏感性。

## 2. 基准与延误结果

基准下，三个目的地两个端点的NY–NJ已证明份额上下界均为零。

| 目的地 | 集中单波次下的精确状态变化 |
|---|---|
| Chicago | 264 h仍为零；288 h两个端点均达到实质使用 |
| Memphis | 276 h仍为零；288 h两个端点均达到实质使用 |
| Columbus最低成本端 | 共同网格内168 h为零；240 h达到32.66% |
| Columbus最短时间端 | 240 h为零；264 h达到32.66% |

以上只表示相邻已测试条件，不是连续现实阈值。共同网格外的Columbus 216 h补充观测为10.056%，满足5%和10%判据，但不满足20%判据。

## 3. NY–NJ的精确价值

| 条件 | 端点 | NY–NJ份额 | 避免成本 | 避免时间 |
|---|---|---:|---:|---:|
| Chicago 288 h | 最低成本端 | 64.89% | 173.2万美元 | 5.030 h |
| Chicago 288 h | 最短时间端 | 32.66% | 226.0万美元 | 3.266 h |
| Memphis 288 h | 最低成本端 | 32.66% | 137.0万美元 | 11.311 h |
| Memphis 288 h | 最短时间端 | 32.66% | 196.6万美元 | 2.613 h |
| Columbus 216 h | 最低成本端 | 10.056% | 14.4万美元 | −1.668 h |
| Columbus 240 h | 最低成本端 | 32.66% | 82.7万美元 | 2.764 h |

成本和时间来自端点特定方案，不得解释为另一个端点方案同时取得两项收益。

首次实质使用时，NY–NJ只挽回0.38%—3.04%的成本损失和0—1.13%的时间损失；到336 h，挽回比例增加至成本10.91%—28.46%、时间7.37%—11.87%。因此，高门户份额不等于挽回大部分中断损失。

在禁止NY–NJ的重点条件中，中断成本损失主要由迟到成本构成：Chicago 288 h、Memphis 288 h和Columbus 240 h的迟到损失分别为5104.9万、5019.6万和3892.2万美元，对应总损失为5700.9万、5608.4万和4502.9万美元。

## 4. 罚金机制

Memphis 288 h最低成本端的NY–NJ份额为：

| 罚金倍率 | 0 | 0.25 | 0.5 | 1 | 1.5 | 2 |
|---|---:|---:|---:|---:|---:|---:|
| NY–NJ份额 | 10.21% | 0 | 22.60% | 32.66% | 32.66% | 32.66% |

份额虽然非单调，但总迟到量与按基准批次罚金率加权的迟到暴露均随罚金倍率上升而非增。0→0.25倍时，四个NY–NJ批次（227.5 FEU）回到更早的西海岸连接；0.25→0.5倍时，十个批次（503.5 FEU）转向NY–NJ并进一步减少迟到。因此，该现象是离散路径与班期配置排序变化，不是求解错误。

使用匹配的零罚金无扰动参照时，Chicago 288 h、Memphis 288 h和Columbus 240 h的成本损失挽回比例分别为91.00%、11.05%和99.38%。由于这三个无扰动参照本身已使用NY–NJ，这些数值是费用设定敏感性，不是中断激活比例。

## 5. 容量位置效应

- 西海岸门户节点容量单独降至35%仍不激活NY–NJ。
- 西海岸区域内陆公路与铁路日容量同时降至50%或30%，在零附加延误下也不激活NY–NJ。
- 50%区域内陆运力与168 h延误叠加时，三个目的地仍为零。
- 30%区域内陆运力与168 h延误叠加时，仅Columbus使用5.724%，低于10%实质使用判据；该条件避免71,787美元和0.538 h。
- Columbus该条件下，洛杉矶—长滩公路连接在模型期29、31达到450 FEU/day，铁路连接在期29达到300 FEU/day；西雅图—塔科马公路连接在期30、31达到450 FEU/day。弧代码仅保留在补充材料。

因此可以写：门户节点容量下降和无附加延误的区域内陆运力下降未激活NY–NJ；容量限制与延误叠加会放大跨海岸响应。不能写成“所有容量扰动都不起作用”。

## 6. 释放设计与偏好

- Chicago最低成本端在全部释放设计中首次于288 h达到实质使用。
- Chicago最短时间端由集中单波次的288 h推迟至所有多波次实例的336 h。
- Memphis多波次大多由288 h推迟至336 h。
- Columbus对释放结构最敏感。

由于释放时刻、需求集中度、资源竞争和可衔接服务日期同时变化，结论只能写成“门户选择对需求时间分布敏感”，不能声称已经单独识别班期相位效应。

ε约束结果表明：Chicago在部分严格交付时间上限下，禁止NY–NJ模型不可行，而允许模型可行；Columbus在相同时间上限下两组模型均可行，但允许NY–NJ可以降低成本。这里的可行性结论只适用于额外时间要求，不能写成NY–NJ对核心网络可行性不可或缺。

## 7. 最终主结论

> 在所构造的三个独立目的地案例中，替代门户的采用不是由单一容量阈值决定。门户节点容量、区域或局部内陆运力、航次延误、目标偏好以及需求时间分布共同决定NY–NJ是否进入最优配置。NY–NJ在严重延误下能够降低成本或时间、扩展可达到的交付时间区域，但首次激活时通常只挽回有限比例的中断损失。

## 8. 最新权威数据文件

- `results/v21_full_structural_core_exact/exact_endpoint_summary.csv`
- `analysis/v21_full_structural_core_exact/disruption_loss_recovery.csv`
- `results/v22_full_structural_cost_sensitivity/exact_endpoint_summary.csv`
- `analysis/v24_full_structural_release_robustness/combined_release_evidence.csv`
- `analysis/v26_full_structural_inland_bottleneck/inland_capacity_delay_interaction.csv`
- `analysis/v27_full_structural_epsilon_tradeoffs/epsilon_tradeoff_summary.csv`
- `analysis/v28_regional_inland_capacity/regional_inland_capacity_comparison.csv`
- `analysis/v28_regional_inland_capacity/binding_regional_inland_arc_days.csv`
- `analysis/v29_memphis_penalty_switching/penalty_level_summary.csv`
- `analysis/v29_memphis_penalty_switching/adjacent_penalty_switches.csv`
- `analysis/v29_memphis_penalty_switching/tardiness_monotonicity_check.csv`
- `analysis/v30_final_results_addenda/zero_penalty_loss_recovery.csv`
- `analysis/v30_final_results_addenda/restricted_disruption_loss_cost_components.csv`
