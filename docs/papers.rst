References
==========

Academic papers underlying PrefGraph's implementation, organized by the methods they enable.
Chapter numbers throughout the documentation refer to Chambers & Echenique (2016).

- Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.


Consistency Testing
-------------------

GARP and HARP test whether budget choices are rationalizable: GARP via SCC + Floyd-Warshall on
the expenditure graph (Varian 1982); HARP for homothetic preferences via log-space Floyd-Warshall
(Varian 1983); Production GARP on profit graphs (Varian 1984). On discrete menus, SARP and
Congruence use Floyd-Warshall on the item graph (Richter 1966).

- Varian, H. R. (1982). The nonparametric approach to demand analysis. *Econometrica*, 50(4), 945–973.
  `[DOI] <https://doi.org/10.2307/1912771>`__
- Varian, H. R. (1983). Non-parametric tests of consumer behaviour. *Review of Economic Studies*, 50(1), 99–110.
  `[DOI] <https://doi.org/10.2307/2296957>`__
- Varian, H. R. (1984). The nonparametric approach to production analysis. *Econometrica*, 52(3), 579–597.
  `[DOI] <https://doi.org/10.2307/1913466>`__
- Richter, M. K. (1966). Revealed preference theory. *Econometrica*, 34(3), 635–645.
  `[DOI] <https://doi.org/10.2307/1909773>`__


Efficiency Measures
-------------------

CCEI and utility recovery use Afriat's LP (Afriat 1967). VEI solves a per-observation LP (Varian 1990).
MPI uses Karp's max-mean-weight cycle algorithm (Echenique, Lee & Shum 2011). The Houtman-Maks index
finds the maximum consistent subset via greedy FVS (Houtman & Maks 1985). The Swaps Index uses a
greedy Feedback Arc Set (Apesteguia & Ballester 2015).

- Afriat, S. N. (1967). The construction of utility functions from expenditure data.
  *International Economic Review*, 8(1), 67–77.
  `[DOI] <https://doi.org/10.2307/2525382>`__
- Varian, H. R. (1990). Goodness-of-fit in optimizing models. *Journal of Econometrics*, 46(1–2), 125–140.
  `[DOI] <https://doi.org/10.1016/0304-4076(90)90051-T>`__
- Echenique, F., Lee, S., & Shum, M. (2011). The money pump as a measure of revealed preference violations.
  *Journal of Political Economy*, 119(6), 1201–1223.
  `[DOI] <https://doi.org/10.1086/665011>`__
- Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent with revealed preference.
  *Kwantitatieve Methoden*, 19, 89–104.
- Apesteguia, J., & Ballester, M. A. (2015). A measure of rationality and welfare.
  *Journal of Political Economy*, 123(6), 1278–1310.
  `[DOI] <https://doi.org/10.1086/683838>`__


Stochastic & Attention
-----------------------

The RUM LP tests whether choice frequencies are consistent with any random utility model via an LP
over K! orderings (Block & Marschak 1960; Kitamura & Stoye 2018). WARP-LA models limited attention
via consideration sets: WARP violations are explained by consumers not seeing all options
(Masatlioglu, Nakajima & Ozbay 2012). Regularity and IIA testing follows Debreu (1960).

- Block, H. D., & Marschak, J. (1960). Random orderings and stochastic theories of responses.
  In I. Olkin et al. (Eds.), *Contributions to Probability and Statistics* (pp. 97–132). Stanford University Press.
- Kitamura, Y., & Stoye, J. (2018). Nonparametric analysis of random utility models.
  *Econometrica*, 86(6), 1883–1909.
  `[DOI] <https://doi.org/10.3982/ECTA14478>`__
- Masatlioglu, Y., Nakajima, D., & Ozbay, E. Y. (2012). Revealed attention.
  *American Economic Review*, 102(5), 2183–2205.
  `[DOI] <https://doi.org/10.1257/aer.102.5.2183>`__
- Debreu, G. (1960). Review of R. D. Luce, *Individual Choice Behavior*. *American Economic Review*, 50, 186–188.


Welfare & Extensions
---------------------

CV/EV welfare measures use expenditure function duality (Vartia 1983). GAPP tests consistency of
preferences over price vectors rather than bundles (Deb, Kitamura, Quah & Stoye 2023). Intertemporal
analysis recovers discount factor bounds via interval propagation (Echenique, Imai & Saito 2020).

- Vartia, Y. O. (1983). Efficient methods of measuring welfare change and compensated income in terms of
  ordinary demand functions. *Econometrica*, 51(1), 79–98.
  `[DOI] <https://doi.org/10.2307/1912249>`__
- Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2023). Revealed price preference: Theory and
  empirical analysis. *Review of Economic Studies*, 90(2), 707–743.
  `[DOI] <https://doi.org/10.1093/restud/rdac041>`__
- Echenique, F., Imai, T., & Saito, K. (2020). Testable implications of models of intertemporal choice.
  *American Economic Journal: Microeconomics*, 12(4), 114–143.
  `[DOI] <https://doi.org/10.1257/mic.20180284>`__


Algorithmic Methods
--------------------

The complexity and greedy algorithms for HM and VEI come from Smeulders et al. (2014). Exact HM
computation via MILP runs in Rust using HiGHS (Demuynck & Rehbeck 2023). The GARP SCC decomposition
follows Talla Nobibon et al. (2015). VEI exact computation follows Mononen (2023). Utility recovery
via Bellman-Ford follows Shiozawa (2016).

- Smeulders, B., Cherchye, L., De Rock, B., & Spieksma, F. C. R. (2014). Goodness-of-fit measures for
  revealed preference tests: Complexity results and algorithms.
  *ACM Transactions on Economics and Computation*, 2(1), Art. 3.
  `[DOI] <https://doi.org/10.1145/2505941>`__
- Demuynck, T., & Rehbeck, J. (2023). Computing revealed preference goodness-of-fit measures with
  integer programming. *Economic Theory*, 75, 1101–1130.
  `[DOI] <https://doi.org/10.1007/s00199-022-01431-7>`__
- Talla Nobibon, F., Smeulders, B., & Spieksma, F. C. R. (2015). A note on GARP testing in one pass.
  *Journal of Optimization Theory and Applications*, 166(3), 1080–1093.
  `[DOI] <https://doi.org/10.1007/s10957-014-0634-1>`__
- Mononen, L. (2023). Computing and comparing measures of rationality. Working paper.
- Shiozawa, K. (2016). Revealed preference test and shortest path problem.
  *Journal of Mathematical Economics*, 67, 1–14.
  `[DOI] <https://doi.org/10.1016/j.jmateco.2016.09.001>`__


AI & Alignment Applications
-----------------------------

Revealed preference methods applied to LLM decision-making and AI alignment. Chen et al. (2023)
tested GPT on GARP/CCEI budget tasks (PNAS). Wen et al. (2025) found that specialization
increases GARP violations. Ge, Procaccia, Halpern et al. (2024) axiomatize alignment from
human feedback using welfare economics. Zhi-Xuan & Carroll (2024) challenge the preference-maximization
framing. Gu & Han (2025) measure divergence between stated and revealed preferences in LLMs.
GARP-EFM (2026) uses revealed preference structure to improve foundation models.

- Chen, Y., Liu, T.-X., Shan, Y., & Zhong, S. (2023). The emergence of economic rationality of GPT.
  *Proceedings of the National Academy of Sciences*, 120(51), e2316205120.
  `[DOI] <https://doi.org/10.1073/pnas.2316205120>`__
- Wen, S. (2025). Economic rationality under specialization: Evidence of decision bias in AI agents.
  Working paper.
- Ge, L., Procaccia, A. D., Vorobeychik, Y., Halpern, D., & Micha, E. (2024).
  Axioms for AI alignment from human feedback. arXiv:2405.12164.
  `[arXiv] <https://arxiv.org/abs/2405.12164>`__
- Zhi-Xuan, T., Carroll, M., Franklin, M., & Ashton, H. (2024). Beyond preferences in AI alignment.
  *Philosophical Studies*.
  `[DOI] <https://doi.org/10.1007/s11098-024-02129-2>`__
- Gu, Z., & Han, S. (2025). Alignment revisited: Are large language models consistent in stated and
  revealed preferences? Working paper.
- GARP-EFM (2026). GARP-EFM: Improving foundation models with revealed preference structure. arXiv.
