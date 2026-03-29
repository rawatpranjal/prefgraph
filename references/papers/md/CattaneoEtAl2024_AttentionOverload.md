Matias D. Cattaneo †

## Attention Overload ∗

Paul H.Y. Cheung ‡

Xinwei Ma §

September 15, 2024

## Abstract

We introduce an Attention Overload Model that captures the idea that alternatives compete for the decision maker's attention, and hence the attention that each alternative receives decreases as the choice problem becomes larger. Using this nonparametric restriction on the random attention formation, we show that a fruitful revealed preference theory can be developed and provide testable implications on the observed choice behavior that can be used to (point or partially) identify the decision maker's preference and attention frequency. We then enhance our attention overload model to accommodate heterogeneous preferences. Due to the nonparametric nature of our identifying assumption, we must discipline the amount of heterogeneity in the choice model: we propose the idea of List-based Attention Overload, where alternatives are presented to the decision makers as a list that correlates with both heterogeneous preferences and random attention. We show that preference and attention frequencies are (point or partially) identifiable under nonparametric assumptions on the list and attention formation mechanisms, even when the true underlying list is unknown to the researcher. Building on our identification results, for both preference and attention frequencies, we develop econometric methods for estimation and inference that are valid in settings with a large number of alternatives and choice problems, a distinctive feature of the economic environment we consider. We provide a software package in R implementing our empirical methods, and illustrate them in a simulation study.

Keywords: attention frequency, limited and random attention, revealed preference, partial identification, high-dimensional inference.

∗ We thank Chris Chambers, Emel Filiz-Ozbay, Whitney Newey, Erkut Ozbay, and seminar participants at Bristol, Cornell, LSE, MIT, Pittsburgh, Princeton, Rice, Rutgers, UC Davis, UCLA, and conference participants at the 2020 Risk, Uncertainty &amp; Decision Conference, the 2020 Southern Economic Association Annual Meeting, the 2022 Cowles Foundation Econometrics Conference, and 2023 Science of Decision Making for their comments. Cattaneo gratefully acknowledges financial support from the National Science Foundation (SES-2241575).

† Department of Operations Research and Financial Engineering, Princeton University.

‡ Jindal School of Management, University of Texas at Dallas

§ Department of Economics, UC San Diego.

¶ Department of Economics, University of Maryland.

Yusufcan Masatlioglu ¶

## 1 Introduction

This paper studies decision making in settings where the decision makers confront an abundance of options, their consideration sets are random, and their attention span is limited. We assume that the attention any alternative receives will (weakly) decrease as the number of rivals increases, a nonparametric restriction on the attention rule of decision makers, which we called Attention Overload . If attention was deterministic, our proposed behavioral assumption would simply say that if a product grabs the consumer's consideration in a large supermarket, then it will grab her attention in a small convenience store as there are fewer options (Reutskaja and Hogarth, 2009; Visschers, Hess, and Siegrist, 2010; Reutskaja, Nagel, Camerer, and Rangel, 2011; Geng, 2016).

Our baseline choice model has two components: a random attention rule and a homogeneous preference ordering, but we later enhance our model to allow for heterogeneous preferences. The random attention rule is the probability distribution on all possible consideration sets. To introduce our attention overload assumption formally, we define the amount of attention a product receives as the frequency it enters the consideration set, termed Attention Frequency . Attention overload then implies that the attention frequency should not increase as the choice set expands. For preferences, we assume that the decision makers have a complete and transitive (initially homogeneous, later heterogeneous) preferences over the alternatives, and that they pick the best alternative in their consideration sets. In this general setting where attention is random and limited, and products compete for attention, we aim to elicit compatible preference orderings and attention frequencies solely from observed choices.

Existing random attention models cannot capture, or are incompatible with, attention overload. For example, Manzini and Mariotti (2014) consider a parametric attention model with independent consideration where each alternative has a constant attention frequency even when there are more alternatives, and therefore their model does not allow decision makers to be more attentive in smaller decision problems. Aguiar (2017) also share the same feature of constant attention frequency. On the other hand, recent research has tried to incorporate menu-dependent attention frequency (Demirkan and Kimya 2020) under the framework of independent consideration. This model is so general that it allows for the opposite behavior than attention overload (i.e., being more attentive in larger choice sets). The recent models of Brady and Rehbeck (2016) and Cattaneo, Ma,

Masatlioglu, and Suleymanov (2020) also allow for the possibility that an alternative receives less attention even when the choice set gets smaller. See Section 2.3 and the supplemental appendix (Section SA.1) for more discussion on related literature.

We contribute to the decision theory literature by introducing the attention overload nonparametric restriction on the attention frequency, which is the key building block to achieve both preference ordering and attention frequency (point or partial) identification from observed choice data. Our results do not require the attention rule to be observed, nor to satisfy other restrictions beyond those implicitly imposed by attention overload. Since our revealed preference and attention elicitation results are derived from nonparametric restrictions on the consideration set formation, without committing to any particular parametric attention rule, they are more robust to misspecification biases (Matzkin, 2007, 2013; Molinari, 2020).

The fact that attention is not observed poses unique challenges to both identification of and statistical inference on the decision maker's preference. This is because one can only identify (and consistently estimate) the choice probabilities from typical choice data, while our main restriction is imposed on the attention rule. Furthermore, as our attention overload assumption does not require a parametric model of consideration set formation, the set of compatible attention rules is usually quite large. In other words, the attention rule is almost never uniquely identified in our model. We nonetheless show that our attention overload assumption, despite being very general, still delivers nontrivial empirical content: we prove in Section 2 that a preference ordering is compatible with our attention overload model if and only if the choice probability satisfies a system of inequality constraints, which corresponds to a form of regularity violation. Furthermore, to improve computation and practical implementation, we discuss how to leverage binary choice problems for identification, estimation, and inference.

Besides revealed preference, information about attention frequency is also an object of interest. For example, it enables marketers to gauge the effectiveness of their marketing strategies, or policymarkers to assess whether consumers allocate their attention to better products. Despite the fact that the underlying attention rule may not be identifiable, we show in Section 2 that our nonparametric attention overload behavioral assumption allows for (point or partial) identification of the attention frequency using standard choice data. This result appears to be the first nonparametric

identification result of a relevant feature of an attention rule in the random limited attention literature: revealed attention analysis has not been possible under nonparametric identifying restrictions in prior work.

In Section 3, we enhance our attention overload model to accommodate multiple decision makers with heterogeneous preference orderings. We begin by providing a general characterization of the partially identified set of distributions on preference orderings and deterministic attention rules. Due to the nonparametric nature of the identifying assumptions, however, the resulting identified set is arguably too large to be useful in practice. Thus, we then further discipline the amount of heterogeneity in the model to (almost) point identify the distribution of preference orderings: we propose the idea of List-Based Attention Overload , where alternatives are presented to the customers as a list that correlates with both heterogeneous preferences and random limited attention. Many real-life situations involve consumers encountering alternatives in the form of a list (Simon, 1955; Rubinstein and Salant, 2006). Our restricted heterogeneous preference model is motivated by the observation that an item's placement on a list has a profound impact on its recollection and evaluation by subjects (Ellison and Ellison, 2009; Augenblick and Nicholson, 2016; Biswas, Grewal, and Roggeveen, 2010; Levav, Heitmann, Herrmann, and Iyengar, 2010). For example, a ranked list of search results provided by a web platform can affect both the search behavior and the perception of individuals about the quality of products (Reutskaja, Nagel, Camerer, and Rangel, 2011).

Our underlying idea is that a common set of characteristics among the decision makers is taken into account to construct the list. For example, a list emerging from search results can be a good proxy for their preferences if decision makers perceive that the search result reflects the true quality of the listed items (Westerwick, 2013). Indeed, many commercial websites collect individual consumers' behavioral data and try to match each consumer with specific products. The list can be thought of as the outcome of personalized recommendations, and therefore individuals facing the same list would share similar tastes. However, individuals might tend to favor their status quo and assign a relatively higher rank to their reference point compared to the rest of the items in the original list. Thus, the existence of the list allows for both heterogeneous preferences and random attention, but restricts the total number of potential preference orderings and attention rules allowed in the choice model.

Attention overload implies that it is often impractical for decision makers to conduct exhaustive searches when many products are on the list. We thus assume that a decision maker investigates alternatives to construct her limited attention consideration set through the list: she might consider only a subset of the alternatives available. Our list-based attention overload model imposes three basic behavioral restrictions on the consideration set formation for a given list: (i) whenever an alternative is considered, all alternatives in the list before it are also taken into account; (ii) if an alternative is not recognized in a smaller set, then it cannot be recognized in a larger set; and (iii) in binary problems, both options are always considered. These assumptions are, for example, supported by eye-tracking studies showing that people tend to scan search engine results in order of appearance, and then fixate on the top-ranked results even if lower-ranked results are more relevant (Pernice, Whitenton, Nielsen, et al., 2018). To capture heterogeneity in cognitive ability, the model accommodates individuals with different consideration sets as long as they satisfy the above behavioral restrictions, thereby allowing for list-based heterogeneity in random limited attention.

To discipline the amount of preference heterogeneity, we also introduce three behavioral axioms characterizing our proposed heterogeneous (preference) attention overload model for a given list. The first axiom captures a restricted form of regularity violation: removing alternatives will not decrease the choice probabilities of a product as long as there is another product listed before it in both decision problems. The second axiom states that binary choice probabilities decrease as the opponent is ranked higher in the list. The last axioms requires that the total binary choice probabilities against the immediate predecessor in the list must be less than or equal to one. We then show that preference and attention frequencies are (point or partially) identifiable under nonparametric assumptions on the list and attention formation mechanisms, even when the true underlying list is unknown to the researcher.

Based on our identification results, covering both homogeneous and heterogeneous preferences settings, we develop econometric methods for revealed preference and attention analysis in both homogeneous and heterogeneous preference settings, which are directly applicable to standard choice data. We only assume that a random sample of choice problems and choices selections is observed, and then provide methods for estimation of and inference on the preference ordering (homogeneous

case), or the preferences frequency (heterogeneous case), and the attention frequency of the decision makers. For example, our methods allow for (i) test whether a specific preference ordering is compatible with our attention overload model, (ii) construct (asymptotically) valid confidence sets, (iii) conduct overall model specification testing, and (iv) estimate preference frequencies in heterogeneous settings. For revealed attention, we obtain (point or partial) identification estimates for attention frequencies in both homogeneous and heterogeneous attention overload models. To establish the validity of our econometric methods, we employ the latest results on high-dimensional normal approximation (Chernozhukov, Chetverikov, Kato, and Koike, 2022). This is crucial because the number of inequality constraints involved in our statistical inference procedures may not be small relative to the sample size. While allowing the dimension (complexity) of the inference problems to be much larger than the sample size, we explicitly characterize the error from a normal approximation for the estimated choice probabilities, thereby shedding light on the finite-sample performance of our proposed econometric methods.

Econometric methods based on revealed preference theory have a long tradition in economics and many other social and behavioral sciences. See Matzkin (2007, 2013), Molinari (2020), and references therein. There is only a handful of recent studies bridging decision theory and econometric methods by connecting discrete choice and limited consideration. Contributions to this new research area include Abaluck and Adams (2021), Barseghyan, Coughlin, Molinari, and Teitelbaum (2021), Barseghyan, Molinari, and Thirkettle (2021), Cattaneo, Ma, Masatlioglu, and Suleymanov (2020), and Dardanoni, Manzini, Mariotti, and Tyson (2020), among others. Each of these papers imposes different (parametric) identification assumptions on the random consideration and the preferences, producing different levels of identification of preference orderings and consideration set rules. We contribute to this emerging literature with new nonparametric results on the identification of and inference for the preference and attention frequencies when decision makers only pay attention to a subset of possibly too many alternatives at random.

The rest of the paper proceeds as follows. Section 2 introduces the setup and our key attention overload assumption under homogeneous preferences, and then proves our main characterization result. That section also presents computationally attractive identification results based on binary comparisons, discusses partial identification of the attention frequency, and outlines valid economet-

ric methods for revealed preference and revealed attention analyses with homogeneous preferences. Section 3 introduces the idea of list-based attention overload to allow for heterogeneous preferences, and presents (point or partial) identification of preference and attention frequencies. That section considers settings where the underlying true list may or may not be known, and also discusses principled econometric methods for estimation and inference using only observed choice data. The appendix contains the proofs of our results, while the supplemental appendix collects (i) in-depth related literature discussion, (ii) simulation evidence, and (iii) omitted technical lemmas and proof details. We also provide a software package and replication code in R implementing our empirical methods.

## 2 Choice under Attention Overload

The theoretical analysis in this section revolves around the assumption that attention frequency is monotonic and preferences are homogeneous. Then, in Section 3, we enhance our choice model to allow for heterogeneous preferences. We denote the grand alternative set as X , and its cardinality by | X | . A typical element of X is denoted by a . We let D be a collection of non-empty subsets of X representing the collection of choice problems. In this section, we allow incomplete data where D is a strict subset of all non-empty subsets of X , which makes the model still applicable when there is missing data. A choice rule is a map π : X ×D → [0 , 1] such that π ( a | S ) = 0 for all a / ∈ S and ∑ a ∈ S π ( a | S ) = 1 for all S ∈ D . π ( a | S ) represents the probability that the decision maker chooses alternative a from the choice problem S . We assume that the choice rule is identifiable from data; that is, it is known or estimable for the purpose of learning about features of the underlying data generating process in econometrics language (Section 2.4).

An important feature of our model is that consideration sets can be random. An attention rule is a map µ : 2 X ×D → [0 , 1] such that µ ( T | S ) = 0 for all T ̸⊆ S and ∑ T ⊆ S µ ( T | S ) = 1 for all S ∈ D . µ ( T | S ) represents the probability of paying attention to the consideration set T ⊆ S when the choice problem is S . This formulation also allows for deterministic attention rules (e.g., µ ( S | S ) = 1 represents full attention). The choice rule and attention rule are standard features of (rational) choice models with random attention. In this paper, we consider a novel feature of these models that is related to the amount of attention each alternative captures for a given µ .

We can extract this information from the attention rule by simply summing up the frequencies of consideration sets containing the alternative.

Definition 1 (Attention Frequency) . Given µ , the attention frequency map ϕ µ : X ×D → [0 , 1] is ϕ µ ( a | S ) := ∑ T ⊆ S : a ∈ T µ ( T | S ) .

ϕ µ ( a | S ) represents the total probability that a attracts attention in S . Whenever µ is clear from the context, we will omit the subscript µ to reduce notation. In deterministic attention models, the attention that one alternative receives is either zero or one (i.e., whether it is being considered or not). However, in stochastic environments, attention is probabilistic: this means that the attention one alternative receives may not be binary.

When decision makers are overwhelmed by an abundance of options, every choice alternative competes for attention. This implies that as the number of alternatives increases, the competition gets more fierce: the attention frequency to a product should decrease weakly when the set of available alternatives is expanded by adding more options. We call this property Attention Overload , the novel nonparametric identifying restriction in this paper.

Assumption 1 (Attention Overload) . For any a ∈ T ⊆ S , ϕ µ ( a | S ) ≤ ϕ µ ( a | T ).

If we allow the consideration set to be empty, then we should also require that the frequency of paying attention to nothing increases when the choice set expands. This is related to the choice overload behavioral phenomenon. At this point, we exclude the possibility of paying attention to nothing for simplicity (i.e., µ ( ∅| S ) = 0 for all S ∈ D ). An attention rule µ satisfies attention overload if its corresponding attention frequency is monotonic in the sense of Assumption 1. Section 2.3 compares and contrasts Assumption 1 with other choice models in the literature (see also Section SA.1 in the supplemental appendix).

Given the nonparametric attention overload restriction in Assumption 1, the choice rule can be defined accordingly. A (rational) decision maker who follows the attention overload choice model maximizes her utility according to a preference ordering ≻ under each realized consideration set.

Definition 2 (Attention Overload Representation) . A choice rule π has an attention overload representation if there exists a preference ordering ≻ over X and an attention rule µ satisfying

attention overload (Assumption 1) such that π ( a | S ) = ∑ T ⊆ S ✶ ( a is ≻ -best in T ) · µ ( T | S ) for all a ∈ S and S ∈ D . In this case, π is represented by ( ≻ , µ ) or, equivalently, π is an Attention Overload Model (AOM). In addition, ≻ represents π if there exists an attention rule µ satisfying attention overload such that π is represented by ( ≻ , µ ).

To summarize, the unknown model primitives are the attention rule µ and the preference ordering ≻ . We only assume that the choice rule π is observable (i.e., point identifiable and estimable from data), and we do not require additional information beyond standard observable choice data. We next investigate the behavioral implications of AOM. Section 2.1 shows that it is possible to (point or partially) identify the underlying preference ordering by exploiting the attention overload Assumption 1, and Section 2.2 presents (point or partial) identification results for the attention frequency. Section 2.4 builds on those identification results and develops feasible econometric methods. Section 3 further demonstrates how to incorporate and study heterogeneous preferences in the presence of attention overload.

## 2.1 Behavioral Implications and Revealed Preferences

We first investigate characterization and preference elicitation, as they have a close relationship. We aim to determine whether a data generating process possesses an AOM representation, and if it is feasible to identify preference orderings from observed choice data. To accomplish this, we investigate whether a specific preference ordering can accurately represent the data. This is a challenging task due to several potential issues. First, the data may not have an AOM representation at all. Second, even if an AOM representation exists, the actual preference may differ from the proposed preference ordering. Third, even if the proposed preference aligns with the underlying preference ordering, it is still necessary to construct an attention rule that satisfies attention overload and accurately represents the data. Our first main result addresses these challenges by providing a tight representation without the requirement of constructing an attention rule, which can be a laborious task when there are many alternatives.

AOM has several behavioral implications. Assume that ( ≻ , µ ) represents π . Since attention is a requirement for a choice, any choice probability is always bounded above by attention frequency, i.e., ϕ ( a | S ) ≥ π ( a | S ). Then, by attention overload, we must have ϕ ( a | T ) ≥ ϕ ( a | S ) ≥ π ( a | S ) for T ⊆ S .

In addition, the difference ϕ ( a | T ) -π ( a | T ) captures the probability that a receives attention but is not chosen in T . As a consequence, in these cases, a better option must be chosen in T , which implies ϕ ( a | T ) -π ( a | T ) ≤ π ( U ≻ ( a ) | T ), where U ≻ ( a ) denotes the strict upper contour set of a with respect to ≻ . (With a slight abuse of notation, we set π ( U ≻ ( a ) | T ) = π ( U ≻ ( a ) ∩ T | T ) = ∑ b ∈ T : b ≻ a π ( b | T ).) Combining these observations, we get π ( a | S ) ≤ ϕ ( a | S ) ≤ ϕ ( a | T ) ≤ π ( U ⪰ ( a ) | T ), where U ⪰ ( a ) denotes the upper contour set of a with respect to ≻ . It follows that π ( a | S ) ≤ π ( U ⪰ ( a ) | T ) whenever ≻ represents the data. This condition only refers to preferences, not to the attention rule. Therefore, the following axiom must be satisfied whenever ≻ represents the data.

Axiom 1 ( ≻ -Regularity) . For all a ∈ T ⊆ S , π ( U ⪰ ( a ) | T ) ≥ π ( a | S ).

Axiom 1 applies to the choice rule π , which is point identifiable and estimable from standard choice data and is stated in terms of a preference ordering ≻ , a key unobservable primitive of our model. Given ≻ , it is routine to check whether π satisfies ≻ -Regularity. This axiom is closely related to, but different from, the classical regularity condition. Axiom 1 trivially implies the regularity condition for the best alternative a ∗ in T , as U ⪰ ( a ∗ ) = { a ∗ } and π ( U ⪰ ( a ∗ ) | T ) = π ( a ∗ | T ) ≥ π ( a ∗ | S ). Hence, the full power of regularity is assumed. For other alternatives, the regularity condition is partially relaxed. At the other extreme, ≻ -Regularity does not restrict the choice probabilities for the worst alternative, a ∗ , since U ⪰ ( a ∗ ) = X , and hence π ( U ⪰ ( a ∗ ) | T ) = 1 ≥ π ( a ∗ | S ) for all T , implying that ≻ -Regularity holds trivially. The following result shows that ≻ -Regularity is not only necessary but also a sufficient condition for ≻ to represent the data.

Theorem 1 (Characterization) . π has an AOM representation with ≻ if and only if π satisfies ≻ -Regularity.

An immediate corollary is that π is AOM if and only if there exists ≻ such that ≻ -Regularity is satisfied. We provided above the proof of the necessity of ≻ -Regularity. The proof of sufficiency, which relies on Farkas's Lemma, is given in the appendix. ≻ -Regularity informs us whether π has an AOM representation with ≻ . Of course, it is possible that ≻ -Regularity can be violated for ≻ but is satisfied for another preference ≻ ′ . Hence, ≻ -Regularity allows us to identify all possible preference orderings without constructing the underlying attention rule µ .

We now turn to the discussion of revealed preference. Option b is revealed to be preferred

to option a if b ≻ a for all ≻ representing π . Theorem 1 suggests that ≻ -Regularity could be a handy method to identify the underlying preference. We first define a binary relation based on ≻ -Regularity property:

<!-- formula-not-decoded -->

Again, P π does not require the construction of all AOM representations. Given a candidate preference, finding the corresponding attention rule satisfying attention overload could be a daunting task. On the other hand, checking whether π satisfying ≻ -Regularity is straightforward because ≻ -Regularity does not require finding the underlying attention rule. Indeed, we utilize this fact in Section 2.4 to develop econometric methods. The next result states the revealed preference of this model.

Corollary 1 (Revealed Preference) . Let π be an AOM. Then, b is revealed preferred to a if and only if b P π a .

Although Theorem 1 and Corollary 1 bypass the need of constructing the underlying attention rule for revealed preference analysis, checking all possible | X | ! preference orderings can be computationally expensive. In addition, if the analyst is interested in learning the preference between two alternatives, a and b , then some constraints suggested by ≻ -Regularity may provide little to no relevant information. Fortunately, a key observation is that regularity violations at binary choice problems can reveal the decision maker's preference. Although this result does not exhaust the nonparametric identification power of Assumption 1, it can be handy and computationally more attractive.

More specifically, if a, b ∈ S and π ( a | S ) &gt; π ( a |{ a, b } ), then any ( ≻ , µ ) representing π must rank b above a , hence b must be preferred to a . To reach such a conclusion, assume the contrary: there exists ( ≻ , µ ) representing π such that a ≻ b and µ satisfying attention overload. First, the attention frequency is always greater (or equal) than the choice probability for any alternative and in any choice set: π ( a | S ) ≤ ϕ ( a | S ). In addition, they are equal for the best alternative in any choice set: a is ≻ -best in S implies π ( a | S ) = ϕ ( a | S ). Given a ≻ b , we have ϕ ( a |{ a, b } ) = π ( a |{ a, b } ) &lt; π ( a | S ) ≤ ϕ ( a | S ). This contradicts our attention overload assumption. The next proposition formalizes this

observation.

Proposition 1 (Regularity Violation at Binary Comparisons) . Let π be an AOM with ( µ, ≻ ) and a, b ∈ S . If π ( a | S ) &gt; π ( a |{ a, b } ), then b ≻ a .

Proposition 1 provides a guideline to easy-to-implement revealed preference analysis without knowledge about each particular representation. A natural question is whether we can generalize the implication of Proposition 1 for an arbitrary set T ⊆ S instead of only for binary sets. The answer is not straightforward because identification from regularity violation may not be as sharp when there are more than two alternatives in the smaller set. From Proposition 1, we are able to claim revealed preference between two alternatives, but when the smaller set contains more than two alternatives, we only know there are some alternatives better than a in the smaller set. To see this, suppose not, and a is the best alternative in the smaller set. We must have π ( a | T ) = ϕ ( a | T ). Given that ϕ ( a | T ) = π ( a | T ) &lt; π ( a | S ) ≤ ϕ ( a, S ), it contradicts the attention overload assumption. We put this observation in the following proposition.

Proposition 2 (Regularity Violation at Bigger Choice Problems) . Let π be an AOM with ( µ, ≻ ) and a ∈ T ⊂ S . If π ( a | S ) &gt; π ( a | T ), then there exists b ∈ T such that b ≻ a .

Both Proposition 1 and 2 are based on regularity violations, and Proposition 2 implies Proposition 1 when T is a binary set. The following example demonstrates how these propositions can be used to limit possible preferences orderings first in order to then apply Theorem 1: regularity violation alone does not exhaust the nonparametric identification power of attention overload in this example.

Example 1. Consider the following choice data:

| π ( ·&#124; S )   | a    |   b | c   | d    |
|-------------------|------|-----|-----|------|
| { a, b, c,d }     | 0.05 | 0.1 | 0.1 | 0.75 |
| { a, b, c,d }     | 0.8  | 0.2 | 0   | -    |
| { a,b, c,d }      | -    | 0.7 | 0.3 | 0    |
| { a, b, c,d }     | 0.9  | 0.1 | -   | -    |

where there are 4! = 24 candidate preference orderings ≻ . First, by applying Proposition 1 from { a, b, c } to { a, b } , we know that any ≻ with b ≻ a would not be represented by the model. Therefore, there are only 12 compatible preferences with AOM. Second, two other regularity violations involve non-binary sets. From { a, b, c, d } to { a, b, c } , since c violate regularity, by Proposition 2, we know that c ≻ a and c ≻ b would not hold simultaneously, which eliminates 4 additional preference candidates. Analogously, from { a, b, c, d } to { b, c, d } , since d violates regularity, it is impossible that d is preferred to both b and c , which further eliminate 4 preferences. Only four preference orderings remain: a ≻ b ≻ c ≻ d , a ≻ b ≻ d ≻ c , a ≻ c ≻ b ≻ d , and a ≻ c ≻ d ≻ b . Finally, we can apply Theorem 1: by checking Axiom 1, we can see that ≻ must include b ≻ d and c ≻ d . This eliminates two preferences. Therefore, the only two possible candidates are a ≻ b ≻ c ≻ d and a ≻ c ≻ b ≻ d .

In the above example, we show that (i) the data has multiple AOM representations, (ii) a ≻ 1 b ≻ c ≻ 1 d and a ≻ 2 c ≻ 2 b ≻ 2 d represent the data, and (iii) since only ≻ 1 and ≻ 2 satisfy Axiom 1, the revealed preference P π is only missing information on b and c (otherwise it is complete). While we had 24 possible candidates, Proposition 1 implied only 12 of them were viable candidates. Then, Proposition 2 eliminated 8 of the remaining ones. Finally, only two satisfied Axiom 1. Hence, Example 1 demonstrates how our main results can be used constructively to identify the set of plausible preferences, while also substantially reducing the computational burden.

## Attentive at Binaries

We established how revealed preferences analysis can be done with our nonparametric attention overload assumption. Due to the nature of attention overload, one might suspect that the decision makers are more likely to pay full attention when there are only two alternatives. We now assume that extreme limited attention (i.e., considering only a single option) at binaries cannot exceed a preset probability level, and investigate its implications for revealed preference.

Assume that for any η ≥ 0 . 5,

<!-- formula-not-decoded -->

This condition puts an upper limit on the magnitude of extreme limited attention. As η increases, the probability of limited attention can increase: η = 1 imposes no constraint on attention behavior. This assumption does not impose a lower bound for full attention. Even in the extreme case, where η is equal to 0 . 5, it is still possible that there is no full attention ( µ ( { a, b }|{ a, b } ) = 0). Hence, it is not a demanding condition when compared to full attention at binaries.

Condition (2) can generate additional revealed preferences: a P B b if π ( a |{ a, b } ) &gt; η , for any η ≥ 0 . 5. Whenever we observe π ( a |{ a, b } ) &gt; η , the choice probability of a cannot be entirely attributed to the attention on the singleton set, µ ( { a }|{ a, b } ) ≤ η . Then, we must have µ ( { a, b }|{ a, b } ) &gt; 0 and the decision maker chooses a over b when she pays attention to both alternatives. Hence, it implies that a must be better than b .

We could also interpret the parameter η as a measure of how cautious the policy-maker is when making a welfare judgment. If η = 1, the policy-maker would not draw any conclusion from binary comparisons only (P B = ∅ ). The choice η = 0 . 5 is commonly used in the literature (Marschak, 1959; Fishburn, 1998), which would refer to the largest P B in our setup-almost uniquely identified.

Condition (2) provides additional revealed preference information if the data on binary comparisons are available. Under (2), the revealed preferences of our model must include P B . We can then extend our characterization theorem: if π satisfies ≻ -Regularity where ≻ includes P B , then the data has an AOM representation with µ satisfying (2). More importantly, P B improves the result of Proposition 2 (and hence Proposition 1) by restricting the set of plausible preferences. We revisit Example 1 to illustrate this point.

Example 1. (continued) Assume that we observe additional data at { b, c } : π ( b |{ b, c } ) = 0 . 7. Then, by Condition (2), we conclude that the only possible preference consistent with the observed choice data is a ≻ b ≻ c ≻ d whenever η ∈ [0 . 5 , 0 . 7), thereby achieving point identification of the preference ordering.

## 2.2 Revealed Attention

Our attention overload model builds on the simple nonparametric requirement that each alternative gets weakly less attention in bigger choice problems, which is captured by monotonicity in attention frequency (Assumption 1). Given a dataset, one might want to learn how the attention frequency

changes across different alternatives and choice problems. For example, marketers might want to gauge the effectiveness of their marketing strategies, or policy markers could be interested in assessing whether consumers allocate their attention to better products. Since we do not put any restriction on the attention rule, the attention frequency can vary depending on the actual attention rule that the decision maker has. This section shows that it is possible to develop upper and lower bounds for the attention frequency and thus achieve partial identification of ϕ .

Consider bounding ϕ from below first. For any superset R ⊇ S , the attention overload assumption implies that π ( a | R ) ≤ ϕ ( a | R ) ≤ ϕ ( a | S ). Therefore, for any S , ϕ ( a | S ) ≥ max R ⊇ S π ( a | R ). This lower bound on the attention frequency only uses information from the choice rule, which is estimable from standard choice data. Importantly, this lower bound does not require a particular AOM representation, that is, it does not require knowledge of the underlying attention rule. It is also possible to derive an upper bound for ϕ , although in this case the bound will depend on the preference ordering. Consider a preference ≻ and an attention rule µ satisfying attention overload, so that π is an AOM with ( ≻ , µ ). Then, for any subset T ⊆ S , ϕ ( a | S ) ≤ ϕ ( a | T ) ≤ π ( U ⪰ ( a ) | T ), which implies that ϕ ( a | S ) ≤ min T ⊆ S π ( U ⪰ ( a ) | T ). These observations give the following theorem.

Theorem 2 (Revealed Attention) . Let π be an AOM and ( µ, ≻ ) represent π . Then, for every a and S such that a ∈ S , max R ⊇ S π ( a | R ) ≤ ϕ µ ( a | S ) ≤ min T ⊆ S : a ∈ T π ( U ⪰ ( a ) | T ).

We now consider three extreme cases of Theorem 2. If both the lower bound and the upper bound are 1, then a attracts full attention at S (Revealed Full Attention). If both bounds are zero, then a does not attract any attention at S (Revealed Inattention). The third case happens when the lower bound is zero and the upper bound is one (No Revealed Attention). Indeed, these three cases are the only possibilities when the data is deterministic, which was studied by Lleras, Masatlioglu, Nakajima, and Ozbay (2017). However, they did not provide any characterization result for revealed attention. Theorem 2 provides such characterization not only for stochastic choice but also for its deterministic counterpart, and hence our theorem is also a novel contribution in the competing attention framework for deterministic choice theory.

Since the stochastic data is richer, Theorem 2 covers another interesting case, which we call Partial Revealed Attention: the upper bound is strictly below one and/or the lower bound is strictly above zero. To illustrate revealed attention, we revisit Example 1.

Example 1. (continued) Given the two possible preferences, we can have bounds on attention frequency. Here, we focus on the choice set { a, b, c, d } . By applying Theorem 2, ϕ ( a |{ a, b, c, d } ) must be 0.05, ϕ ( b |{ a, b, c, d } ) and ϕ ( c |{ a, b, c, d } ) must be between 0 . 1 and 0 . 25 and ϕ ( d |{ a, b, c, d } ) must be between 0 . 75 and 1.

In some cases, the attention frequency will be uniquely identified for certain alternatives. For instance, in addition to Example 1, we have π ( c | R ) = 0 . 25 for some R ⊇ { a, b, c, d } . Then, the lower bound for ϕ ( c |{ a, b, c, d } ), while being free of the underlying preference, must be 0 . 25. Hence, the attention frequency is point-identified to be 0 . 25 since the upper bound is also 0 . 25.

Theorem 2 is useful in real world applications to inform a firm/government how much attention each product/policy receives among other options. While the lower bound can be interpreted as the pessimistic evaluation for attention, the upper bound captures optimistic evaluation. The question is whether these local pessimistic (optimistic) evaluations hold globally, that is, we ask whether there is an underlying attention rule µ satisfying attention overload such that the attention frequencies agree with the pessimistic (optimistic) evaluations for every set. Due to the richness in attention rule allowed by our Assumption 1, it turns out that the answer is affirmative.

Theorem 3 (Pessimistic Evaluation for Attention) . Let π be an AOM and ( ≻ , µ ) represent π . Then there exists a pessimistic attention rule µ ∗ such that ( ≻ , µ ∗ ) is also an AOM representation of π . That is, for all S , ϕ µ ∗ ( a | S ) = max R ⊇ S π ( a | R ).

This theorem concerns the pessimistic evaluation case, but an analogous result can be established for the optimistic evaluation. As a consequence, in econometrics language, Theorem 2 delivers the sharp identified set for ϕ .

## 2.3 Comparison to Other Random Attention Models

We compare AOM to other existing (random) attention models: Tversky (1972), Manzini and Mariotti (2014), Brady and Rehbeck (2016), Aguiar (2017), Cattaneo, Ma, Masatlioglu, and Suleymanov (2020), and Demirkan and Kimya (2020). With the exception of Cattaneo, Ma, Masatlioglu, and Suleymanov (2020), which imposes a nonparametric restriction, all the other models introduce the idea of random limited attention with a parametric restriction on the attention rule. We show

that none of these models can capture the attention overload assumption by comparing their underlying attention rules. Section SA.1 in the supplemental appendix provides further comparisons between AOM and the related literature.

Consider two individuals, Ann and Ben. In a larger decision problem, S , Ann pays attention to all alternatives with probability one (full attention, µ Ann ( S | S ) = 1), while Ben experiences attention overload and focuses only on a single alternative a in S while ignoring the rest (limited attention, µ Ben ( { a }| S ) = 1). We chose these two extreme cases to make our point clear. Existing evidence suggests that, as the size of the available options decreases, the phenomenon of choice overload becomes less evident, leading decision makers to overlook less alternatives. Hence, assuming | T | &lt; | S | , Ann continues to exhibit full attention in T but Ben considers more alternatives in T . Table 1 summarizes the comparisons with the literature using these two decision makers.

Table 1: Predictions of each model for a smaller set T for Ann and Ben, given that they exhibit full attention and extremely limited attention in the larger set, respectively.

|                                                                                                                                                               | Ann                                                                                            | Ben                                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                               | Larger set S                                                                                   | Larger set S                                                                                                                   |
|                                                                                                                                                               | Full attention µ Ann ( S &#124; S ) = 1                                                        | Limited attention µ Ben ( { a }&#124; S ) = 1                                                                                  |
|                                                                                                                                                               | Predictions for a smaller set a ∈ T ⊂ S                                                        | Predictions for a smaller set a ∈ T ⊂ S                                                                                        |
| Manzini and Mariotti (2014) Tversky (1972); Aguiar (2017) Brady and Rehbeck (2016) Cattaneo, Ma, Masatlioglu, and Suleymanov (2020) Demirkan and Kimya (2020) | µ Ann ( T &#124; T ) = 1 µ Ann ( T &#124; T ) = 1 No restriction No restriction No restriction | µ Ben ( { a }&#124; T ) = 1 µ Ben ( { a }&#124; T ) = 1 µ Ben ( { a }&#124; T ) = 1 µ Ben ( { a }&#124; T ) = 1 No restriction |
| Attention Overload Model (AOM)                                                                                                                                | µ Ann ( T &#124; T ) = 1                                                                       | Increasing Attention                                                                                                           |

The Independence Attention Model (IAM) of Manzini and Mariotti (2014) and 'Elimination by Aspects' Model (EAM) (Tversky, 1972; Aguiar, 2017) make the same predictions for Ann and Ben. Manzini and Mariotti (2014) considers a parametric model of limited attention where each alternative has a constant attention frequency. Full attention on the larger selection implies that the attention frequency for each alternative in S is one. (Some of the parametric models require the attention parameters be strictly between zero and one, but we can capture these examples either by allowing the parameters to be equal to zero and one, or by taking a limit.) 'Elimination by

Aspects' attention is an adaptation of Tversky (1972) into limited attention, where alternatives are exogeneously bundled into categories, and the decision maker considers each of these categories with certain probabilities. Aguiar (2017) characterizes a special case of this model with the default option where each category includes the default option. For Ann, both models make the same prediction, which is consistent with attention overload: Ann should pay full attention in T . However, these models do not allow Ben to be more attentive for smaller sets: Ben must pay attention to the singleton { a } with probability one. In IAM, this is because the attention frequencies for other alternatives in S are zero, while in EAM, a is the only alternative belonging to the most popular category. Hence, Manzini and Mariotti (2014) and (Tversky, 1972; Aguiar, 2017) are too restrictive to accommodate attention overload. Furthermore, Demirkan and Kimya (2020) drops the menuindependence assumption in IAM, which leads to no restriction on the attention rule for T , and hence cannot accommodate ateention overload either.

The Logit Attention Model (LAM) of Brady and Rehbeck (2016) and the Random Attention Model (RAM) of Cattaneo, Ma, Masatlioglu, and Suleymanov (2020) make the same prediction for our individuals. LAM could be interpreted as a parametric limited attention model where each subset could be the consideration set with some probability. Since Ann exhibits full attention in the larger set, S must be the most probable consideration set, and since S is not a subset of T , Ann is allowed to focus on any subset of T , including a single alternative. For Ben, this model implies Ben must continue to pay attention only to a . RAM imposes a (nonparametric) monotonicity assumption on the attention rule, so that µ ( T | S ) ≤ µ ( T | S \ b ) for every b / ∈ T ⊆ S . In RAM, the monotonic attention does not impose any restriction on Ann's behavior (for example, it could be µ Ann ( { a }| T ) = 1), but Ben must pay attention to the same single alternative ( µ Ben ( { a }| T ) = 1). In other words, Ann can exhibit the opposite of choice overload, while Ben must exhibit limited attention even in a smaller problem. Therefore, these two models stand in contrast to the concept of attention overload.

In contrast to the existing literature, our AOM introduces a novel nonparametric assumption on attention frequency, which measures how much attention each alternative (rather than each consideration set) receives. That is, ϕ ( b | S ) ≤ ϕ ( b | S \ c ) for every b ∈ S . If Ann behaves according to AOM, µ Ann ( S | S ) = 1 implies ϕ Ann ( b | S ) = 1 for all b , which dictates ϕ Ann ( b | T ) = 1 for all b ∈ T (full

attention). For Ben, AOM imposes that he must consider a for sure ( ϕ Ben ( a | T ) = 1), but attention frequency for other alternatives can increase. In particular, AOM allows that ϕ Ben ( b | T ) = 1 for all b ∈ T (full attention). This discussion makes it clear that our attention overload property is distinct from all other (parametric or nonparametric) random limited attention models. As a consequence, our AOM captures novel empirical findings and describes novel attention allocation behaviors compared to the existing models in the literature.

## 2.4 Econometric Methods

We obtained several testable implications and related results for the AOM: Theorem 1, Propositions 1 and 2, and Theorem 2. Our next goal is to develop econometric methods to implement these findings using real data, which can help elicit preferences, conduct empirical testing of our AOM, and provide confidence sets for attention frequencies. To this end, we rely on a random sample of observations consisting of choice data for n units indexed by i = 1 , 2 , . . . , n . Each unit faces a choice problem Y i , and her choice is denoted by y i ∈ Y i . This is formally stated in the assumption below. We recall that D ⊆ 2 X \ ∅ is a collection of choice problems.

Assumption 2 (Choice Data) . The data consists of a random sample of choice problems and observed choices { ( Y i , y i ) : 1 ≤ i ≤ n } with Y i ∈ D and P [ y i = a | Y i = S ] = π ( a | S ).

A given preference ordering ≻ is compatible with our AOM if and only if ≻ -Regularity holds. In other words, each preference ordering corresponds to a collection of inequality constraints, which we collect in the following hypothesis:

<!-- formula-not-decoded -->

To construct a test statistic for testing the hypotheses in (3), we can replace the unknown choice probabilities by their estimates, ̂ π ( a | S ) and ̂ π ( U ⪰ ( a ) | T ), leading to ̂ D ( a | S, T ) = ̂ π ( a | S ) -̂ π ( U ⪰ ( a ) | T ). For example, ̂ π ( a | S ) = 1 N S ∑ n i =1 ✶ ( Y i = S, y i = a ), with N S = ∑ n i =1 ✶ ( Y i = S ). We define the following test statistic

<!-- formula-not-decoded -->

where ̂ σ ( a | S, T ) is the standard error of ̂ D ( a | S, T ), that is, ̂ σ 2 ( a | S, T ) = 1 N S ̂ π ( a | S ) ( 1 -̂ π ( a | S ) ) + 1 N T ̂ π ( U ⪰ ( a ) | T ) ( 1 -̂ π ( U ⪰ ( a ) | T ) ) . The outer max operation in T ( ≻ ) guarantees that we will never reject the null hypothesis if none of the estimated differences ̂ D ( a | S, T ) are strictly positive. In other words, a preference is not ruled out by our analysis if ≻ -Regularity holds in the sample. The statistic depends on a specific preference ordering which we would like to test against for: such dependence is explicitly reflected by the notation T ( ≻ ).

We investigate the statistical properties of the test statistic in order to construct valid inference procedures, building on the recent work of Chernozhukov, Chetverikov, and Kato (2019) and Chernozhukov, Chetverikov, Kato, and Koike (2022) for many moment inequality testing. (See also Molinari (2020) for an overview and further references.) We seek for a critical value, denoted by cv( α, ≻ ), such that under the null hypothesis (i.e., when the preference is compatible with our AOM), P [ T ( ≻ ) &gt; cv( α, ≻ )] ≤ α + r ≻ , where α ∈ (0 , 1) denotes the desired significance level of the test, and r ≻ denotes a quantifiable error of approximation (which should vanish in large samples with possibly many inequalities).

To provide some intuition on the critical value construction, the Studentized test statistic, ̂ D ( a | S, T ) / ̂ σ ( a | S, T ), is approximately normally distributed with mean D ( a | S, T ) /σ ( a | S, T ). Since D ( a | S, T ) ≤ 0 under the null hypothesis, the above normal distribution will be first-order stochastically dominated by the standard normal distribution. Letting ̂ D be the column vector collecting all ̂ D ( a | S, T ), and Ω be its correlation matrix, then our test statistic T ( ≻ ) will be dominated by the maximum of a normal vector with a zero mean and a variance of Ω , up to the error from normal approximation. Using properties of Bernoulli random variables, an estimate of Ω can be constructed with the estimated choice probabilities and the effective sample sizes. We denote the estimated correlation matrix by ̂ Ω . The critical value is then the (1 -α )-quantile of the maximum of a Gaussian vector: cv( α, ≻ ) = inf { t : P [ T G ( ≻ ) ≤ t | Data] ≥ 1 -α } with T G ( ≻ ) = max { max( ̂ Ω 1 / 2 z ) , 0 } , where the inner max operation computes the maximum over the elements of ̂ Ω 1 / 2 z , and z denotes a standard normal random vector of suitable dimension. Precise definitions and omitted details are given in the supplemental appendix to conserve space. The theorem below offers formal statistical guarantee on the validity of our proposed test.

Theorem 4 (Preference Elicitation with Theorem 1) . Assume Assumption 2 holds. Let c 1 be the

number of comparisons (i.e., inequalities) in (3), and c 2 = ( min S ∈D N S ) · ( min a ∈ T ⊂ S T,S ∈D σ ( a | S, T ) ) . Then, under the null hypothesis H 0 in (3), P [ T ( ≻ ) &gt; cv( α, ≻ )] ≤ α + r ≻ with r ≻ = C · ( log 5 ( n c 1 ) / c 2 2 ) 1 / 4 , where C denotes an absolute constant.

This theorem shows that the error in distributional approximation, r ≻ , only depends on the dimension of the problem (i.e., c 1 ) logarithmically, and therefore our estimation and inference procedures remain valid even if the test statistic T ( ≻ ) involves comparing 'many' pairs of estimated choice probabilities. By providing non-asymptotic statistical guarantees, our procedures can accommodate situations where both the number of alternatives and the number of choice problems are large, and hence they are expected to perform well in finite samples, leading to more robust welfare analysis results and policy recommendations.

It is routine to incorporate condition (2) into our econometric implementation. Specifically, the test statistic and the critical value we introduced above are based on moment inequality testing. To accommodate the new assumption on attentive at binaries, one only needs to include additional probability comparisons corresponding to the η -constrained revealed preference.

Given the testing procedures we developed, it is easy to construct valid confidence sets by test inversion. To be precise, a dual asymptotically valid 100(1 -α )% level confidence set is CS (1 -α ) = {≻ : T ( ≻ ) ≤ cv( α, ≻ ) } . Therefore, for any preference ≻ that is compatible with our AOM, we have the statistical guarantee on coverage: P [ ≻ ∈ CS (1 -α )] ≥ 1 -α -r ≻ .

Revealed preference between two alternatives, a and b , can be analyzed based on Theorem 1 by checking, say, if a ≻ b for all identified preferences ≻ in CS (1 -α ). This approach to revealed preference has the advantage that it exhausts the identification power of our AOM. On the other hand, it comes with a nontrivial computational cost as discussed in Section 2.1. Therefore, we also discuss how Proposition 1 can be implemented in practice. The econometric methods stemming from Theorem 1 and Proposition 1 are complementary: while the former provides a more systematic framework for preference revelation, the latter can be handy if binary comparisons are available in the data or if the analyst is particularly interested in inferring preference ordering among pairs of alternatives. (Also see Example 1, which demonstrates that applying Proposition 1 first to a choice data may greatly reduce the number of preference orderings to be tested against ≻ -Regularity.)

We fix two alternatives, say a and b , and let D ab be the collection of choice problems containing both a and b , excluding the binary comparison; that is, D ab = { S ∈ D : S ⊋ { a, b }} . We also recall the simplified notation π ( a | b ) = π ( a |{ a, b } ) for binary comparisons. Then, we may deduce the preference ordering b ≻ a if we are able to reject H 0 : max S ∈D ab D ( a | S, b ) ≤ 0, where we define D ( a | S, b ) = π ( a | S ) -π ( a | b ). Constructing a test statistic is straightforward:

<!-- formula-not-decoded -->

where ̂ D ( a | S, b ) = ̂ π ( a | S ) -̂ π ( a | b ), and ̂ σ ( a | S, b ) is its standard error. We employ the same technique to construct a critical value. Letting ̂ D be the column vector collecting all ̂ D ( a | S, b ), and ̂ Ω be its estimated correlation matrix. The critical value is cv( α, ab ) = inf { t : P [ T G ( ab ) ≤ t | Data] ≥ 1 -α } with T G ( ab ) = max { max( ̂ Ω 1 / 2 z ) , 0 } . The next result follows from Theorem 4.

Corollary 2 (Preference Elicitation with Proposition 1) . Assume Assumption 2 holds. Let c = ( min { min S ∈D ab N S , N { a,b } }) · ( min S ∈D ab σ ( a | S, b ) ) . Then, under the null hypothesis that H 0 : max S ∈D ab D ( a | S, b ) ≤ 0, the size distortion satisfies P [ T ( ab ) &gt; cv( α, ab )] ≤ α + r ab with r ab = C · ( log 5 ( n |D ab | ) / c 2 ) 1 / 4 , where C denotes an absolute constant.

One of our main econometric contributions is a careful study of the properties of the estimated correlation matrix, ̂ Ω , where we provide an explicit bound on the supremum of the entry-wise estimation error ∥ ̂ Ω -Ω ∥ ∞ . This is further combined with the results in Chernozhukov, Chetverikov, Kato, and Koike (2022) to establish a normal approximation for the centered and normalized inequality constraints, say ( ̂ D ( a | S, T ) -D ( a | S, T )) / ̂ σ ( a | S, T ). Cattaneo, Ma, Masatlioglu, and Suleymanov (2020) considered preference elicitation under a monotonic attention assumption, and proposed estimation and inference procedures based on pairwise comparison of choice probabilities as in (3). However, their econometric analysis assumed 'fixed dimension,' and hence did not allow the complexity of the problem to be 'large' relative to the sample size, which is required in this paper.

We now discuss how to operationalize the partial identification result in Theorem 2 on attention frequency. We will illustrate with the lower bound, ϕ ( a | S ) ≥ max R ⊇ S π ( a | R ), since the upper bound follows analogously. A na¨ ıve implementation would replace the unknown choice probabilities

by their estimates. Unfortunately, the uncertainty in the estimated choice probabilities will be amplified by the maximum operator, leading to over-estimated lower bounds. Our aim is to provide a construction of the lower bound, denoted as ̂ ϕ ( a | S ) such that P [ ϕ ( a | S ) ≥ ̂ ϕ ( a | S )] ≥ 1 -α + r ϕ ( a | S ) , with α ∈ (0 , 1) denoting the desired significance level, and r ϕ ( a | S ) denoting the error in approximation, which should become smaller as the sample size increases.

Our construction is based on computing the maximum over a collection of adjusted empirical choice probabilities. To be precise, we define

<!-- formula-not-decoded -->

where ̂ σ ( a | R ) = √ ̂ π ( a | R )(1 -̂ π ( a | R )) /N R is the standard error of the estimated probability ̂ π ( a | R ), and cv( α, ϕ ( a | S )) = inf { t : P [max( z ) ≤ t ] ≥ 1 -α } , with z is a standard normal random vector of dimension |{ R ∈ D : R ⊇ S }| , the number of supersets of S .

To provide some intuition for the construction, we begin with the normal approximation: ̂ π ( a | R ) a ∼ Normal( π ( a | R ) , σ ( a | R )). Then, the estimated choice probabilities are mutually independent since they are constructed from different subsamples, and

<!-- formula-not-decoded -->

where ≈ denotes an approximation in large samples. Heuristically, the above demonstrates that with high probability (approximately 1 -α ) the true choice probabilities, π ( a | R ), are bounded from below by ̂ π ( a | R ) -cv( α, ϕ ( a | S )) · ̂ σ ( a | R ). This is made possible by the adjustment term we added to the estimated choices probabilities. This idea is formalized in the following theorem, which offers precise probability guarantees.

Theorem 5 (Attention Frequency Elicitation with Theorem 2) . Let π be an AOM, and assume Assumption 2 holds. Define c 1 = |{ R ∈ D : R ⊇ S }| to be the number of supersets of S , and c 2 = ( min R ⊇ S,R ∈D N R ) · ( min R ⊇ S,R ∈D σ ( a | R ) ) . Then, P [ ϕ ( a | S ) ≥ ̂ ϕ ( a | S )] ≥ 1 -α + r ϕ ( a | S ) with r ϕ ( a | S ) = C · ( log 5 ( n c 1 ) / c 2 2 ) 1 / 4 , where C denotes an absolute constant.

Due to space limitations, the supplemental appendix reports simulation evidence showcasing the empirical performance of our theoretical and methodological econometric results. To be more precise, we test against four preference orderings using simulated data, and in this process, we vary the number of choice problems available ( D ) in the data and the effective sample size of each choice problem ( N S ). Overall, our procedure performs well: it is able to reject preference orderings that are not compatible with ≻ -Regularity with nontrivial power, while at the same time maintaining size.

## 3 Heterogeneous Preferences

The analysis so far considered homogeneous preferences, and hence our choice model assumed that every decision maker has the same taste but different levels of attentiveness. In this section, we present a model that describes the choice behavior at individual and population levels allowing for preference heterogeneity in the underlying data-generating process as well as attention heterogeneity. Our heterogeneous preference choice model can be used empirically with aggregate data on a group of distinct decision makers where each may differ not only on what they pay attention to but also on what they prefer. The model also allows heterogeneous preferences that may correlate with attention, full independence being a special case.

We consider a group of individuals who differ not only in how they pay attention but also in their preferences. We focus on individuals with deterministic choices. To do so, we first define a deterministic attention rule satisfying the attention overload property. An attention rule µ is deterministic if µ ( T | S ) is either 0 or 1. Since µ ( ·| S ) is a probability distribution, there exists a unique subset of S , say T , such that µ ( T | S ) = 1. Hence, we can define a mapping Γ from 2 X to 2 X \ ∅ so that Γ( S ) = T where µ ( T | S ) = 1. Γ( S ) represents the consideration set under S . Our Assumption 1 implies the following condition for the consideration set mapping Γ: If an alternative is recognized in a larger set, it is also recognized in a smaller one; this condition first appeared in Lleras, Masatlioglu, Nakajima, and Ozbay (2017). Formally, if a ∈ Γ( S ) and a ∈ T ⊂ S , then a ∈ Γ( T ). Indeed, this property is equivalent to Assumption 1 in the class of the deterministic attention rules. 1 We denote all consideration sets satisfying Assumption 1 by AO .

1 When µ is deterministic, then ϕ µ ( a | S ) is 1 if a ∈ Γ( S ), otherwise 0. If ϕ µ ( a | S ) = 0, there is no need to check.

We are now ready to define our model allowing heterogeneous preferences. Let P denote the collection of all preferences. Consider a population of individuals where each individual is endowed with two primitives; a preference ordering ≻∈ P and a deterministic consideration set mapping Γ ∈ AO . Each pair (Γ , ≻ ) represents a choice type in the population, whose choices are deterministic. Let τ be a probability distribution on AO×P , so τ (Γ , ≻ ) is the probability of (Γ , ≻ ) being the choice type. 2 Each τ naturally induces a probabilistic choice function:

<!-- formula-not-decoded -->

The 'rational' types are allowed to be on the support of τ (take Γ( S ) = S ). On the other hand, there are many other 'non-rational' choice types. Hence, one might wonder whether this model has any prediction power. It is routine to show this model has empirical content when the number of alternatives is greater or equal to 4. Using an idea from McFadden and Richter (1990), we are able to provide a full characterization for the model.

Theorem 6. A choice rule π is represented by a probability distribution τ over AO × P so that (4) holds if and only if

<!-- formula-not-decoded -->

for any finite sequence { ( a i , S i ) } n i =1 with a i ∈ S i ∈ D .

This theorem provides a method for falsifying the model: a single violated inequality suffices to indicate that the data cannot be accurately represented by this model. 3 However, identifying τ within this model is highly unsatisfactory. For instance, it is possible to construct multiple representations { τ i } such that (i) they all represent the same choice behavior, (ii) they all differ from each other, and (iii) their supports are disjoint. Hence, there is no hope for point identification,

If ϕ µ ( a | S ) = 1, then a must be in Γ( S ), which implies that a must be in Γ( T ). Hence ϕ µ ( a | T ) = 1 satisfying Assumption 1.

2 One could imagine a more general model where each type is a pair of ( µ, ≻ ) where µ is a stochastic attention rule satisfying Assumption 1. Then τ is a probability distribution over ( µ, ≻ ). As we elaborate below, this general model would suffer from non-uniqueness even more.

3 Demonstrating sufficiency, unfortunately, involves an infinite number of linear inequalities. This is also true for the Axiom of Revealed Stochastic Preference (ARSP) of McFadden and Richter (1990), for which Kitamura and Stoye (2018) develop empirically tests thereof.

and little hope for informative partial identification results; even partial identification does not help constrain the parameters to lie in a strict subset of the parameter space. This complexity arises from two sources of non-uniqueness. Firstly, even under full attention, the classical Random Utility Model (RUM) is known to suffer from non-uniqueness due to varying tastes (Fishburn, 1998; Turansick, 2022). Secondly, the issue is exacerbated by the consideration set mapping, which can lead to different consideration sets representing the same deterministic choice, even with fixed preferences. Consequently, without additional structure, developing useful identification results for τ at the present level of generality is arguably a hopeless exercise.

To achieve point identification with preference heterogenity, we impose further nonparametric restriction on the heterogeneity in preference types and possible consideration sets to discipline our proposed choice model: our key idea is that alternatives are presented to the decision maker as a list that correlates with both heterogeneous preferences and random (limited) attention. 4 Our modeling strategy that individuals come across various options presented as a list is motivated by real-world scenarios. For example, customers often browse Amazon's ordered search results or receive a ranked list of advertisements, not to mention decision makers employing web search engines.

We first assume that the list is observable. This assumption may be reasonable in situations where we can observe Amazon's product list for each product category, a ballot for a specific election, Google's search results for a keyword, etc. In Appendix B, we relax this assumption and endogenize the list, allowing for the identification of heterogeneous preferences when the true underlying list is unknown to the researcher. The supplementary appendix (Section SA-1) gives a review of the related literature on limited attention and heterogeneous preferences, and of choice theory over a list.

4 In decision theory, Rubinstein and Salant (2006) is the seminal paper introducing the idea that decision makers encounter the alternatives in the form of a list. This idea has been influential since then (see for example, Horan (2010); Guney (2014); Yildiz (2016); Aguiar, Boccardi, and Dean (2016); Kovach and ¨ Ulk¨ u (2020); Ishii, Kovach, and ¨ Ulk¨ u (2021); Tserenjigmid (2021); Yegane (2022); Koshevoy and Savaglio (2023); Manzini, Mariotti, and ¨ Ulk¨ u (2024)).

## 3.1 List-Based Attention Overload

In this subsection, we propose a model that accommodates varying tastes and attention mechanisms with point identification. Formally, we assume there is a list of items represented by the linear order ▷ . Let ⟨ a 1 , a 2 , . . . , a | X | ⟩ be the enumeration of the elements in X with respect to ▷ , where a j denotes the item in the j th position, and | X | is the size of the grand set X . In other words, j &lt; k implies that a j appears earlier in the list than a k , which is equivalent to saying a j ▷ a k . We will use both notation, j &lt; k and a j ▷ a k , interchangeably. For a choice problem S ⊆ X , we also enumerate its elements as ⟨ a s 1 , a s 2 , . . . , a s | S | ⟩ .

It is often impractical for consumers to conduct exhaustive searches because their attention is limited. Through the list, a decision maker investigates alternatives to construct her consideration set but she might consider only a subset of the alternatives available to her due to limited attention. Our proposed model will impose three basic behavioral restrictions on the consideration set formation for a given list. Let Γ( S ) be the consideration set when the choice problem is S . First, we assume that the consideration set obeys the underlying order: if a k belongs to the consideration set, so does every feasible alternative that appeared before a k , i.e., a k ∈ Γ( S ) and a j ∈ S such that j &lt; k imply a j ∈ Γ( S ). In other words, whenever an alternative is considered, all alternatives in the list before it are also taken into account. Second, we assume that alternatives are the deterministic counterpart of AOM as discussed before. Finally, we assume that each individual always considers both items in binary problems, i.e, Γ( S ) = S whenever | S | = 2.

Definition 3 (List-based Attention Overload) . A deterministic consideration set mapping Γ satisfies list-based attention overload on ▷ (i.e., ⟨ a 1 , a 2 , . . . , a | X | ⟩ ) if (i) for every T ⊆ S , a k ∈ Γ( S ) and a j ∈ T , j ≤ k implies a j ∈ Γ( T ); (ii) Γ( S ) = S whenever | S | = 2.

We denote AO ▷ as the set of all consideration set rules satisfying list-based attention overload with respect to ▷ .

Individuals are also heterogeneous in terms of their preferences. Unlike RUM, we assume that the set of preferences is related to the underlying list. First, our model recognizes that some individuals perceive search results as reflecting the true quality of listed items. Indeed, many commercial websites collect individual consumers' behavioral data and try to match each consumer

with personally relevant products. The list can be thought of as the outcome of personalized recommendations. Individuals facing the same list share similar tastes. However, our model also captures the idea that individuals might favor their status quo, meaning that they assign a relatively higher rank to their reference point compared to other items in the original list. This assumption restricts potential preferences exhibited in the model. For all j &lt; k , define ≻ kj as a linear order where the k th alternative in ▷ is moved to the j th position. To give some examples, ≻ 21 corresponds to the ordering ⟨ a 2 , a 1 , a 3 , a 4 , . . . , a | X | ⟩ , and ≻ 42 is ⟨ a 1 , a 4 , a 2 , a 3 , . . . , a | X | ⟩ . We call ≻ kj a single improvement of ▷ , and we use P ▷ to denote the set of all single improvements of ▷ including ▷ itself. For notational convenience, we let ≻ kk = ▷ . If ≻∈ P ▷ , this implies that there exists a unique alternative a k such that its relative ranking improved with respect to ▷ . Orderings involving multiple changes to the original list order ▷ , such as ⟨ a 2 , a 1 , a 4 , a 3 , . . . , a | X | ⟩ , are not allowed.

Equipped with AO ▷ ×P ▷ , we state the precise definition of the model.

Definition 4 (Heterogeneous Preference Attention Overload) . A probabilistic choice function π has a Heterogeneous Preference Attention Overload representation with respect to ▷ (HAOM ▷ ) if there exists τ on AO ▷ ×P ▷ such that π ( a | S ) = τ ({ (Γ , ≻ ) ∈ AO ▷ ×P ▷ : a is ≻ -best in Γ( S ) }) .

HAOM ▷ introduces heterogeneity both in terms of preferences and attention. This feature makes this model independent of RUM. (If the support of τ consists of only choice types with Γ( S ) = S , then the model becomes a special case of RUM.) Due to limited attention, HAOM ▷ allows choice types outside of the preference maximization paradigm, and therefore it captures behaviors outside of RUM. On the other hand, since RUM allows more preference types, some choice behaviors can be only captured by RUM but not by HAOM ▷ . Having said that, HAOM ▷ still encompasses more choice types than RUM, even though we restrict the set of possible preferences. This is because HAOM ▷ considers two types of heterogeneity (attention and preferences), while RUM only allows for heterogeneity in preference ordering. Our AOM can be regarded as another extreme of HAOM ▷ , as it requires that every choice type has the same preferences but it imposes a mild nonparametric restriction on attention. In the supplemental appendix we discuss further the relationship between AOM and HAOM, and their connections with prior literature.

## 3.2 Characterization

We now provide a list of behavioral postulates describing the implications of HAOM ▷ for a given list ▷ . Since the model is more involved in terms of types, the following results require that D includes all non-empty subsets of X , which is a common assumption on choice models that involve random utility.

We first highlight that HAOM ▷ allows violations of the regularity condition, while regularity always holds in RUM. However, HAOM ▷ limits the types of regularity violations that are permissible. For example, there will be no regularity violation as long as the first alternative in the list is always present. This is because only choice types { Γ : a k ∈ Γ( S ) }×{≻ k 1 } will pick a k in the presence of a 1 (assuming k &gt; 1), and they will continue choosing a k in a smaller choice problem T ⊂ S . Indeed, we can generalize this intuition: removing alternatives will not decrease the choice probabilities of a product as long as there is another product listed before it in both decision problems. Consider two products a k and a j such that j &lt; k , then the choice probability of a k obeys regularity conditions, i.e., π ( a k | T ) ≥ π ( a k | S ) for T ⊂ S provided that a j ∈ T . This is our first behavioral postulate for HAOM ▷ .

Axiom 2 (List-Regularity) . For all a j , a k ∈ T ⊂ S with j &lt; k , π ( a k | T ) ≥ π ( a k | S ).

The following property imposes a structure on binary choices. It simply says everything else equal, being listed earlier increases choice probabilities on binary comparisons. For example, compared to the third product in the list, the first product is chosen more frequently than the second one. In other words, binary choice probabilities decrease as the opponent is ranked higher in the list. For example, a 3 is going to be chosen against a 2 more often than against a 1 . This is because the individual considers both alternatives in every binary comparison. Hence, being listed earlier is reflected in choice probabilities. We adopt the following notation for binary comparisons: π ( a k | a ℓ ) := π ( a k |{ a k , a ℓ } ).

Axiom 3 (List-Monotonicity) . For all a j , a k , a ℓ such that j &lt; k &lt; ℓ , π ( a ℓ | a k ) ≥ π ( a ℓ | a j ).

Again consider binary comparisons. Assume we have π ( a 2 | a 1 ) = 0 . 6. This implies that the frequency of ≻ 21 must be 0 . 6, and those types must prefer a 2 over a 3 as well (our model only allows preferences that are single improvements over the original list order). Hence, π ( a 3 | a 2 ) must

be smaller than 0 . 4 = 1 -0 . 6. The next axiom generalizes this idea: the total binary choice probabilities against the immediate predecessor in the list must be less than or equal to 1.

Axiom 4 (List-Boundedness) . ∑ | X | j =2 π ( a j | a j -1 ) ≤ 1.

HAOM ▷ introduces some compatibility among all the preference types in the population because each preference type is a single improvement of ▷ . However, it allows for a significant level of heterogeneity in attention. Our next theorem indicates that HAOM ▷ can still make predictions because it states that the three axioms completely characterize HAOM ▷ .

Theorem 7 (Characterization) . Given ▷ , a choice rule π satisfies Axioms 2-4 if and only if π has an HAOM ▷ representation.

Theorem 7 establishes both a necessary and sufficient condition for HAOM ▷ . The importance of this theorem lies in its applicability to any dataset, even when RUM is not applicable. This makes it a powerful tool for studying choice behaviors beyond utility maximization. In contrast to RUM, our choice model enjoys strong identification for preferences in P ▷ : the frequency of each preference type in P ▷ is uniquely (point) identified. Formally, we define the preference type frequency for each ≻ , τ ( ≻ ) := τ ( { (Γ , ≻ ) : Γ ∈ AO ▷ } ); that is, τ ( ≻ ) represents the total probability of ≻ being the underlying preferences. If both τ 1 and τ 2 are HAOM ▷ representations of π , then τ 1 ( ≻ ) = τ 2 ( ≻ ). The uniqueness of HAOM ▷ is in sharp contrast to the non-uniqueness of RUM.

Given the uniqueness result, we now ask whether we can reveal the frequency of specific preference types. Our revealed preference serves (at least) two purposes. First, it identifies the specific form of heterogeneity in the population in terms of preferences. Second, it provides a unique weight for each preference type within this heterogeneity. For example, by using these weights, a policymaker can evaluate how a particular policy affects each agent in the heterogeneous population and then combine these effects with precisely identified weights. Furthermore, when we interpret ≻ kj as the status quo bias preferences, τ ( ≻ kj ) is the frequency of people whose default option is a k and the strength of bias is j -k (the difference between the original and final ranking of the default a k ). Hence, our identification result can be used to measure the status quo bias in the data. For the theorem below, we recall the adopted convention that ≻ 11 := ▷ , which corresponds to choice types whose preference coincides with the original list order.

Theorem 8 (Revealed Preference Types) . Let τ be a HAOM ▷ representation of π . Then (i) τ ( ≻ kj ) = π ( a k | a j ) -π ( a k | a j -1 ) for k &gt; j &gt; 1, (ii) τ ( ≻ k 1 ) = π ( a k | a 1 ), and (iii) τ ( ≻ 11 ) = 1 -∑ | X | k =2 π ( a k | a k -1 ) .

To provide some intuition, recall from Definition 3 that decision makers pay full attention in binary comparisons. This implies that for j &lt; k , the choice probability π ( a k | a j ) is just the frequency of choice types who rank a k at or higher than the j th position; that is, π ( a k | a j ) = ∑ ℓ ≤ j τ ( ≻ kℓ ). This observation justifies the first part of the theorem. Now take j = k -1. Then π ( a k | a k -1 ) is the frequency of choice types who do not agree with ▷ on the ranking of a k , which leads to the second conclusion of the theorem.

The identification result provided by Theorem 8 is based on binary choices. While it provides point identification, it can only be used when all binary comparisons { a k , a j } such that k &gt; j are available in the data. When the data is incomplete, in the sense that not all possible comparisons are observed (or identifiable and estimable in econometrics language), we can provide bounds for the frequency of preference types: if a non-top alternative is chosen, it must be attributed to the types who like that alternative better than the top alternative. In this sense, the choice probability is the lower bound of those types since some of these decision maker types might not have looked far enough down the list (i.e., random and limited attention).

Proposition 3 (Bounds for Preference Types) . Let τ be a HAOM ▷ representation of π . Fix S and let a s 1 be its top-listed item. Then, for k &gt; s 1 , π ( a k | S ) ≤ τ ( { (Γ , ≻ ) : Γ ∈ AO ▷ and a k ≻ a s 1 } ).

To close this subsection, we demonstrate how to perform revealed attention analysis on our choice model with heterogeneous preference and random attention; c.f., Section 2.2. Given τ , we define the attention frequency for an alternative ϕ τ ( a | S ) := τ ( { (Γ , ≻ ) : a ∈ Γ( S ) } ); that is, ϕ τ ( a | S ) represents the total probability of a being considered in the population when the choice set is S . Under RUM, this is assumed to be equal to one for all alternatives in S (full attention). In our model, while full attention holds for the first alternative on the list, it might not hold for other alternatives. Even though we cannot point identify the attention frequency for other alternatives, we provide upper and lower bounds. For notation convenience, we will drop the subscript and use ϕ ( a | S ) = ϕ τ ( a | S ).

Theorem 9 (Revealed Attention) . Let τ be a HAOM ▷ representation of π . Fix S and let a s 1 be its top-listed item. Then, (i) ϕ ( a s 1 | S ) = 1; (ii) for a k ∈ S and k &gt; s 1 ,

<!-- formula-not-decoded -->

## 3.3 Econometric Methods

We first discuss how to empirically bound the frequency of different preference types. To implement the partial identification result in Proposition 3, we fix two positions, 1 ≤ j &lt; k . We are interested in bounding the fraction of decision maker who rank a k at or above the j th position. In other words, we consider the following parameter of interest: θ kj = ∑ ℓ ≤ j τ ( ≻ kℓ ) . We have θ kj ≥ θ ks 1 ≥ π ( a k | S ) for any choice problem ⟨ a s 1 , a s 2 , . . . , a s | S | ⟩ satisfying (i) a k ∈ S , and (ii) s 1 ≤ j , where the first inequality follows from the definition of θ kj , and the second inequality follows by Proposition 3.

We construct a lower bound for θ kj by computing the maximum over a collection of adjusted empirical choice probabilities. (The same idea has been used to empirically bound the attention frequency in Section 2.4 and Theorem 5.) Let D kj = { S : a k ∈ S, s 1 ≤ j } . We define

<!-- formula-not-decoded -->

As before, ̂ σ ( a k | S ) = √ ̂ π ( a k | S )(1 -̂ π ( a k | S )) /N S is the standard error of the estimated probability ̂ π ( a k | S ), and the critical value is cv( α, θ kj ) = inf { t : P [ max( z ) ≤ t ] ≥ 1 -α } , where z is a standard normal random vector of dimension |D kj | . The statistical validity of the proposed lower bound is guaranteed by the following theorem.

Theorem 10 (Revealed Preference Distribution with Proposition 3) . Let π be an HAOM ▷ , and assume Assumption 2 holds. Define c = ( min S ∈D kj N S ) · ( min S ∈D kj σ ( a k | S ) ) . Then, P [ θ kj ≥ ̂ θ kj ] ≥ 1 -α + r θ kj , where r θ kj = C · ( log 5 ( n |D kj | ) / c 2 ) 1 / 4 with C an absolute constant.

We now discuss how to bound the attention frequency using the results of Theorem 9. We will illustrate with the lower bound since the upper bound follows analogously. Fix some choice problem S , and some option a k ∈ S which differs from the top-listed one. As before, D is the collection of

choice problems available in the data. We form the lower bound as

<!-- formula-not-decoded -->

where ̂ π ( U ▷ ( a k ) | R ) is the empirical probability of choosing an option listed before a k in R , and ̂ σ ( U ▷ ( a k ) | R ) = √ ̂ π ( U ▷ ( a k ) | R )(1 -̂ π ( U ▷ ( a k ) | R )) /N R is its standard error. The critical value is constructed as cv( α, ϕ ( a k | S )) = inf { t : P [ max( z ) ≤ t ] ≥ 1 -α } , where z is a standard normal random vector of dimension |{ R ∈ D : R ⊇ S }| , the number of supersets of S .

Theorem 11 (Attention Frequency Elicitation with Theorem 9) . Let π be an HAOM ▷ , and assume Assumption 2 holds. Define c 1 = |{ R ∈ D : R ⊇ S }| to be the number of supersets of S , and c 2 = ( min R ⊇ S,R ∈D N R ) · ( min R ⊇ S,R ∈D σ ( B ▷ ( a k ) | R ) ) . Then, P [ ϕ ( a k | S ) ≥ ̂ ϕ ( a k | S )] ≥ 1 -α + r ϕ ( a k | S ) , where r ϕ ( a k | S ) = C · (log 5 ( n c 1 ) / c 2 2 ) 1 / 4 , and C denotes an absolute constant.

To showcase the performance of our proposed econometric methods, we report the results of a simulation study on preference elicitation in the supplemental appendix. Our numerical findings illustrate how the lower bound on preference distribution (i.e., θ kj above) changes as we vary the effective sample size and the number of choice problems available in the data.

## A Appendix: Proofs

Proof of Theorems 1 and 3 . Corollary 1 is implied by Theorem 1, that is, ≻ -Regularity captures all of the empirical content that our AOM delivers for revealed preference. In addition, we already showed the necessity of ≻ -Regularity, and hence we only need to prove sufficiency for Theorem 1. On the other hand, Theorem 3 will be shown in the following proof because we also prove the existence of an attention rule that uses pessimistic evaluation. (The optimistic evaluation follows analogously.)

The proof is divided into two parts. The first part sets up a system of linear equations that pins down the attention rule satisfying the desired property. Some algebraic operations are devoted to lining up the system in preparation for the second part of the proof, which utilizes Farkas's Lemma to prove the existence of a solution to the system of equations for any parameter value which satisfies ≻ -Regularity.

Assume ( π, ≻ ) satisfies property ≻ -Regularity. For every S and x ∈ S , a compatible attention

rule should explain the data, i.e., ∑ x ∈ T ⊆ S x is ≻ -best in T µ ( T | S ) = π ( x | S ). In addition, we would like to set the attention rule such that it gives the pessimistic evaluation, i.e., ϕ ( x | S ) = max R ⊇ S π ( x | R ). (We set ϕ ( x | S ) = max R ⊇ S π ( x | R ), which is the pessimistic evaluation. An alternative proof can use the optimistic evaluation, i.e., ϕ µ ( a | S ) = min T ⊆ S π ( U ⪰ ( a ) | T ), and same proof strategy goes through. In fact, for any attention frequency between these bounds, our proof remains valid if we choose ϕ such that it satisfies attention overload.) If the above is feasible, then the resulting attention rule will satisfy the attention overload assumption. It remains to show that there exists a solution to the system of linear equations. Let x 1 ≻ x 2 ≻ · · · ≻ x n . Then, we have for i = 1 , · · · , n

<!-- formula-not-decoded -->

For x 1 , ≻ -Regularity requires that, for any R ⊇ S , ϕ ( x 1 | S ) ≥ π ( x 1 | S ) ≥ π ( x 1 | R ), which implies max R ⊇ S π ( x 1 | R ) = ϕ ( x 1 | S ) = π ( x 1 | S ). In addition, we also have ∑ x 1 ∈ T ⊆ S x 1 is ≻ -best in T µ ( T | S ) = ∑ x 1 ∈ T ⊆ S µ ( T | S ) = ϕ ( x 1 | S ) = π ( x 1 | S ) . It gives us P 1 = M 1 , which says that the probability that the best alternative is chosen is the attention it received. On the other hand, P n is π ( x n | S ) = µ ( { x n }| S ), which immediately gives the solution to the 'unknown' µ ( { x n }| S ). Hence, we are left with P i for i = 1 , · · · , n -1 and M i for i = 2 , · · · , n . Then, we create M ′ i ≡ ∑ j ≤ i P j -M i for every i = 2 , · · · , n , that is,

<!-- formula-not-decoded -->

The above makes sense because ∑ j ≤ i π ( x j | S ) -max R ⊇ S π ( x i | R ) ≥ 0 for i = 2 , · · · , n , which is required by ≻ -Regularity. Lastly, we define P ′ 1 ≡ P 1 -∑ j&gt; 1 M j . We are left with P ′ 1 , P i for i = 2 · · · , n -1, and M ′ i for i = 2 , · · · , n . We utilize Farkas's Lemma to prove the existence of solution to the above system of linear equations. The system is straightforward when n ≤ 2, so we focus on n ≥ 3.

Lemma A.1 (Farkas' Lemma) . Let A ∈ R m × n and b ∈ R m . Then exactly one of the following is true: (1). There exists an x ∈ R n such that Ax = b and x ≥ 0. (2). There exists a y ∈ R m such that yA ≥ 0 and yb &lt; 0.

We let A be the matrix and b be the vector such that the above system of linear equations is represented by A µ = b . Specifically, A = ( r 1 , r 2 , · · · , r 2 n -2 ) ⊤ , and b = ( b 1 , b 2 , · · · , b 2 n -2 ) ⊤ , where r j 's are column vectors. In particular, we let r 1 and b 1 correspond to the LHS and RHS of P ′ 1 respectively; r j and b j correspond to the LHS and RHS of M ′ n +2 -j respectively for j = 2 , · · · , n ; r j and b j correspond to the LHS and RHS of P -n +1+ j respectively for j = n +1 , · · · , 2 n -2. To save notation, let m i := max R ⊇ S π ( x i | R ), π i := π ( x i | S ) and k i := ∑ j ≤ i π ( x j | S ) -max R ⊇ S π ( x i | R ) =

∑ j ≤ i π j -m i , for all i . Let B be the collection of b subject to the condition ≻ -Regularity:

<!-- formula-not-decoded -->

We show that there does not exist y = ( y 1 , y 2 , y 3 , · · · , y 2 n -2 ) ∈ R 2 n -2 such that yA ≥ 0 and yb &lt; 0 for any b ∈ B . We define the set Y ( A ) as the set of y which satisfies yA ≥ 0. Hence, it suffices to show that for all b ∈ B , min y ∈Y ( A ) yb ≥ 0. Except for b 1 , all the other b j are positive for all possible π ( ·|· ) as long as the choice rule satisfies ≻ -Regularity. The key insight in the following proof is to show how we can guarantee yb ≥ 0 despite the fact the possibility of b 1 being negative.

By construction, A admits a reduced row-echelon form. Since A admits a reduced row-echelon form, the leading entry is 1 and the leading entry in each row is the only non-zero entry in its column. Then, we know that it gives y j ≥ 0 for all j (i.e., for all y ∈ Y ( A ), y ≥ 0). With this observation, we can see that if b 1 ≥ 0, then the proof becomes trivial. Therefore, we assume b 1 &lt; 0. We then further explore the restriction of y under the requirement that yA ≥ 0.

Lemma A.2. For all y ∈ Y ( A ), non-empty P ⊆ { 2 , 3 , · · · , n -1 } , and j = ¯ P +1 , ¯ P +2 , · · · , 2 n -¯ P , we have ∑ i ∈ P y i + y j ≥ | P | · y 1 , where | P | is the cardinality of P , and ¯ P is the largest element in P .

We need to show an auxiliary minimization problem to complete the proof. Let c n and z n be two vectors. To be consistent with the above in notation, both vectors start with subscript 2 and end with 2 n -2. i.e. c n = ( c 2 , c 3 , · · · , c 2 n -2 ) ⊤ .

Lemma A.3. For all n ≥ 3, min c n ∈ C n , z n ∈ Z n c n · z n ≥ 1, where

<!-- formula-not-decoded -->

Z n = { z n ∈ R 2 n -3 + ∣ ∣ ∣ ∑ i ∈ P z i + z j ≥ | P | , ∀ non-empty P ⊆ { 2 , 3 , · · · , n -1 } , j = ¯ P +1 , ¯ P +2 , · · · , 2 n -¯ P }

and ¯ P denotes the largest element in P .

Since b 1 &lt; 0, we can apply Lemma A.3 by setting c i = -b i b 1 and z i = y i y 1 for i = 2 , · · · , 2 n -2. Firstly, Lemma A.2 implies that the set of constraints in Z n is fulfilled in Y ( A ) after we plug in z i = y i y 1 . Secondly, all the constraint in the set C n is fulfilled after we plugged in c i = -b i b 1 , due to the way B is constructed. Therefore, the statement that all b ∈ B , min y ∈Y ( A ) yb ≥ 0 is implied by the statement that min c n ∈ C n , z n ∈ Z n c n · z n ≥ 1. It remains to prove Lemmas A.2 and A.3.

Proof of Lemma A.2. For any set P , we get ∑ i ∈ P y i + y j ≥ | P | · y 1 from the column of µ ( S -∪ i ∈ P ∪{ j } { x i + n -2 }| S ) for any j ∈ { ¯ P + 1 , ¯ P + 2 , · · · , ¯ P + n } . For the LHS: it is because for any i ∈ P , the vector r n -i +2 has the coefficient of 1 in the column of µ ( S - ∪ i ∈ P ∪{ j } { x i + n -2 }| S ) by construction. For the RHS: it is because the vector r 1 has the coefficient of -| P | in the same column by construction. Also, we get ∑ i ∈ P y i + y j ≥ | P | · y 1 from the column of µ ( S - ∪ i ∈ P { x i + n -2 } -∪ i&lt;j -n { x i }| S ) for any j ∈ { n +1 , n +2 , · · · , 2 n -¯ P } . For the LHS: it is because for any i ∈ P , the vector r n -i +2 has the coefficient of 1 in the column of µ ( S -∪ i ∈ P { x i + n -2 } - ∪ i&lt;j -n { x i }| S ) by construction. For the RHS: it is because the vector r 1 has the coefficient of -| P | in the same column by construction. Hence, we have covered any j in { ¯ P +1 , ¯ P +2 , · · · , 2 n -¯ P } , which concludes the proof.

Proof of Lemma A.3. We prove this by induction. Consider n = 3. We have C 3 = { c 3 ∈ R 3 + | c 2 ≥ 1 , c 4 + c 3 ≥ 1 } and Z 3 = { z 3 ∈ R 3 + | z 2 + z 3 ≥ 1 , z 2 + z 4 ≥ 1 } . It is straight-forward to see that, if z 2 ≥ 1, c 3 · z 3 ≥ c 2 z 2 ≥ 1. Therefore, we consider the case that z 2 &lt; 1. Then, we have, by putting in all the constraints, we get c 3 · z 3 = z 2 c 2 + z 3 c 3 + z 4 c 4 ≥ z 2 (1) + c 3 (1 -z 2 ) + c 4 (1 -z 2 ) = z 2 + ( c 3 + c 4 )(1 -z 2 ) ≥ 1. Hence, it is true for n = 3. Therefore, suppose the claim holds for n = k -1, and consider n = k . We set up the Lagrangian minimization problem and assign Lagrangian multipliers λ i to the constraints in C n . For notational convenience, we label each multiplier by all the subscripts involved in the corresponding constraint. Take n = 3 as an example, then we would have multipliers λ 2 for c 2 ≥ 1 and λ 3 , 4 for c 4 + c 3 ≥ 1. It is simple to check that each constraint has it own unique subscript. We collect all possible subscript labels into the set Λ k . (The Lagrangian multiplier for the constraints in Z n is not much used in the proof.) We then get the first order condition of the Lagrangian equation with the complementary slackness conditions:

<!-- formula-not-decoded -->

̸

By the first order condition, we have c k · z k ≥ ∑ 2 k -2 i =2 c i ( ∑ i ∈ I ∈ Λ k λ I ) = ∑ I ∈ Λ k λ I ( ∑ i ∈ I c i ) ≥ ∑ I ∈ Λ k λ I , where the last inequality applies the inequality constraint in C k . If c i = 0 for all i , then ∑ I ∈ Λ k λ I ≥ 1. For example, if c 2 k -2 = 0 and c 2 = 0, we can get binding constraint due to complementary slackness such that z 2 = ∑ 2 ∈ I ∈ Λ k and z 2 k -2 = ∑ 2 k -2 ∈ I ∈ Λ k λ I . Then, apply the respective constraint from Z k , we have z 2 + z 2 k -2 ≥ 1 ⇒ ∑ 2 ∈ I ∈ Λ k λ I + ∑ 2 k -2 ∈ I ∈ Λ k λ I ≥ 1 ⇒ ∑ I ∈ Λ k λ I ≥ 1. In fact, it is straight-forward to check that as long as

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

Then ∑ I ∈ Λ k λ I ≥ 1. For cases outside the above, we check sequentially and apply induction hypothesis in each scenario.

̸

Case 1 : c 2 = 0. Notice that under the specification of c 2 = 0, the set of permissible choice of c k is smaller. Then, we re-number some of the variables. In particular, we write z ′ i = z i +1 and c ′ i = c i +1 for i = 2 , 3 , · · · , 2( k -1) -2. We name this set of relabeled constraint as C k | Case 1 where both c ′ i and c j for some i, j co-exist. In particular, we have now c k = ( c 2 , c 3 , ..., c 2 k -2 ) := (0 , c ′ 2 , c ′ 3 , ..., c ′ 2( k -1) -2 , c 2 k -2 ). We perform the same procedure on Z k . Then, by restricting attention to c ′ i and z ′ i , we can see that C k | Case 1 ⊂ C ′ k -1 and Z k | Case 1 ⊂ Z ′ k -1 , where C ′ k -1 is the same set as C k -1 by just relabeling c to c ′ . Hence, in this case, by induction hypothesis, min c k ∈ C k | Case 1 , z k ∈ Z k | Case 1 c k · z k ≥ min c k -1 ∈ C ′ k -1 , z k -1 ∈ Z ′ k -1 c k -1 · z k -1 ≥ 1. Therefore, if c 2 = 0, the proof is done. Let c 2 = 0. As shown above, if c 2 k -2 = 0, the proof is done. Then, we look into cases where c 2 k -2 = 0. It comes Case 2, where we first assume c 3 = 0.

̸

̸

Case 2 : c 2 = 0, c 3 = 0 and c 2 k -2 = 0. We re-label the variable, in particular, write c ′ i = c i for i = 2, c ′ i = c i +1 for i = 3 , · · · , 2( k -1) -2. Then, we have c k = ( c 2 , c 3 , ..., c 2 k -2 ) := ( c ′ 2 , 0 , c ′ 3 , ..., c ′ 2( k -1) -3 , c ′ 2( k -1) -2 , 0), where 0 ′ s are guaranteed by the case supposition. Analogously, we do the same for z . By a similar argument. We show can min c k ∈ C k | Case 2 , z k ∈ Z k | Case 2 c k · z k ≥ 1. Therefore, if c 3 = 0 and c 2 k -2 = 0, the proof is done. Let c 3 = 0. As shown above, since c 2 = 0, if c 2 k -3 = 0, the proof is done. Thus, on top of the assumption that c 2 k -2 = 0, we look into cases where c 2 k -3 = 0. It comes Case 3, where we assume c 4 = 0.

̸

̸

Case 3 : c 2 , c 3 = 0, c 4 = 0 and c 2 k -3 = c 2 k -2 = 0. We re-label the variable, in particular, write c ′ i = c i for i = 2 , 3, c ′ i = c i +1 for i = 4 , · · · , 2( k -1) -3, and c ′ i = c i +2 for i = 2( k -1) -2. Then, we have c k = ( c 2 , c 3 , ..., c 2 k -2 ) := ( c ′ 2 , c ′ 3 , 0 , c ′ 5 , ..., c ′ 2( k -1) -3 , 0 , c ′ 2( k -1) -2 ), where 0 ′ s are guaranteed by the case supposition (we do not list all of the 0 ′ s in the specification so that one can see easier where the c ′ are defined). Analogously, we relabel z . By a similar argument. We can show that min c k ∈ C k | Case 3 , z k ∈ Z k | Case 3 c k · z k ≥ 1. Continuing this argument, we skip to Case k -2.

̸

Case k -2: c 2 , c 3 , c k -2 = 0, c k -1 = 0 and c k +2 = · · · = c 2 k -3 = c 2 k -2 = 0. Write c ′ i = c i for i = 2 , ..k -2, c ′ i = c i +1 for i = k -1 , k , and c ′ i = c i +2 for i = k +1 , .., 2( k -1) -2. Analogously, we do the same for z . By a similar argument, we can show min c k ∈ C k | Case k-2 , z k ∈ Z k | Case k-2 c k · z k ≥ 1.

̸

Case k -1 (Last Case) : c 2 , c 3 , c k -1 = 0, c k = c k +1 = · · · = c 2 k -2 = 0. Write c ′ i = c i for i = 2 , ..k -1 and c ′ i = c i +2 for i = k, k +1 , .., 2( k -1) -2. Analogously, we do the same for z . By a similar argument, we can show min c k ∈ C k | Case k-1 , z k ∈ Z k | Case k-1 c k · z k ≥ 1.

Therefore, the above covers every possible case and it shows that in every case the minimum is greater than or equal to 1 for n = k . By induction, the minimum of the objective function is

̸

̸

greater than or equal to 1 for all n ≥ 3.

Proof of Theorem 4 . We denote the choice probabilities by the vector π , and the constraints implied by ≻ -Regularity will be collected in the matrix R ≻ . σ contains the standard deviations, that is, σ 2 are the diagonal elements in the covariance matrix R ≻ V [ ˆ π ] R ⊤ ≻ . As before, Ω and ˆ Ω are the true and estimated correlation matrices of R ≻ ˆ π . The operation max( · ) computes the largest element in a vector/matrix. ⊘ denotes Hadamard (element-wise) division. The supremum norm of a vector/matrix is ∥ · ∥ ∞ . Finally, we use c to denote some constant, whose value may differ depending on the context.

The next lemma gives a Gaussian approximation to the infeasible centered and scaled sum.

Lemma A.4. Let ˇ z be a mean-zero Gaussian random vector with the covariance matrix Ω . Then,

<!-- formula-not-decoded -->

The next step is to replace the infeasible standard error.

Lemma A.5. Let ξ 1 , ξ 2 &gt; 0 with ξ 2 → 0. Then, P [ ∥ ( R ≻ ˆ π -R ≻ π ) ⊘ σ ≻ -( R ≻ ˆ π -R ≻ π ) ⊘ ˆ σ ≻ ∥ ∞ ≥ ξ 1 ξ 2 ] is bounded by cξ -1 1 √ log c 1 + c ( log 5 ( n c 1 ) / c 2 2 ) 1 4 + c exp { -1 c c 2 2 ξ 2 2 +log c 1 } .

The proof considers a sequence of approximations, which will rely on the following statistics: T ( ≻ ) = max { ( R ≻ ˆ π ) ⊘ ˆ σ ≻ , 0 } , T ◦ ( ≻ ) = max { ( R ≻ (ˆ π -π )) ⊘ ˆ σ ≻ , 0 } , ˜ T ◦ ( ≻ ) = max { ( R ≻ (ˆ π -π )) ⊘ σ ≻ , 0 } , T G ( ≻ ) = max { z , 0 } , and ˇ T G ( ≻ ) = max { ˇ z , 0 } . We will also define the following quantiles/critical values: cv( α, ≻ ) = inf { t ≥ 0 : P [ T G ( ≻ ) ≤ t | Data] ≥ 1 -α } , ˇ cv( α, ≻ ) = inf { t ≥ 0 : P [ ˇ T G ( ≻ ) ≤ t ] ≥ 1 -α } , and ˜ cv( α, ≻ ) = inf { t ≥ 0 : P [max(ˇ z ) ≤ t ] ≥ 1 -α } .

To show the validity of the above critical value, we first need an error bound on the two critical values, cv( α, ≻ ) and ˇ cv( α, ≻ ). The following lemma will be useful.

Lemma A.6. The critical values, cv( α, ≻ ) and ˇ cv( α, ≻ ), satisfy P [ ˇ cv ( α + cξ 1 2 3 log c 2 , ≻ ) ≤ cv( α, ≻ ) ≤ ˇ cv ( α -cξ 1 2 3 log c 2 , ≻ )] ≥ 1 -c exp { -1 c c 2 2 ξ 2 3 +2log c 1 } .

To wrap up the proof of Theorem 4, we rely on a sequence of error bounds. First notice that, under H 0 , P [ T ( ≻ ) &gt; cv( α, ≻ )] ≤ P [ T ( ≻ ) ◦ &gt; ˇ cv( α + cξ 1 2 3 log c 1 , ≻ )] + c exp {-1 c c 2 2 ξ 2 3 +2log c 1 } . Next, we apply Lemma A.5 and obtain that P [ T ◦ ( ≻ ) &gt; ˇ cv( α + cξ 1 2 3 log c 1 , ≻ )] is bounded by P [ ˜ T ◦ ( ≻ ) &gt; ˇ cv( α + cξ 1 2 3 log c 1 , ≻ ) -ξ 1 ξ 2 ] + cξ -1 1 √ log c 1 + c ( log 5 ( n c 1 ) c 2 2 ) 1 4 + c exp {-1 c c 2 2 ξ 2 2 +log c 1 } . The final error bound in our analysis is due to Lemma A.4, which implies that P [ ˜ T ◦ ( ≻ ) &gt; ˇ cv( α + cξ 1 2 3 log c 1 , ≻ ) -ξ 1 ξ 2 ] is bounded by P [ ˇ T G ( ≻ ) &gt; ˇ cv ( α + cξ 1 2 3 log c 1 , ≻ ) -ξ 1 ξ 2 ] + c ( log 5 ( n c 1 ) c 2 2 ) 1 4 . Collecting all pieces, the

size distortion P [ T ( ≻ ) &gt; cv( α, ≻ )] is bounded by P [ ˇ T G ( ≻ ) &gt; ˇ cv( α + cξ 1 2 3 log c 1 , ≻ ) -ξ 1 ξ 2 ] and the error below:

<!-- formula-not-decoded -->

To proceed, we employ the following anti-concentration result of normal random vectors (Lemma D.4 in the supplemental appendix to Chernozhukov, Chetverikov, and Kato 2019).

Lemma A.7. Let ˇ z ∈ R c 1 be a mean-zero normal random vector such that V [ˇ z ℓ ] = 1 for all 1 ≤ ℓ ≤ c 1 . Then, for any t ∈ R and any ϵ &gt; 0, P [ | max( z ) -t | ≤ ϵ ] ≤ 4 ϵ ( √ 2 log c 1 +1).

First assume ˇ cv( α + cξ 1 2 3 log c 1 , ≻ ) &gt; 0, then ˇ cv( α + cξ 1 2 3 log c 1 , ≻ ) = ˜ cv( α + cξ 1 2 3 log c 1 , ≻ ). By applying Lemma A.7, we have ˜ cv( α + cξ 1 2 3 log c 1 +4 ξ 1 ξ 2 ( √ 2 log c 1 +1) , ≻ ) ≤ ˜ cv( α + cξ 1 2 3 log c 1 , ≻ ) -ξ 1 ξ 2 , which further implies that P [ ˇ T G ( ≻ ) &gt; ˇ cv( α + cξ 1 2 3 log c 1 , ≻ ) -ξ 1 ξ 2 ] is bounded by α + cξ 1 2 3 log c 1 + 4 ξ 1 ξ 2 ( √ 2 log c 1 +1. Finally, we have that P [ T ( ≻ ) &gt; cv( α, ≻ )] is bounded by α plus the following error

<!-- formula-not-decoded -->

To control the above error, we need to verify a few side conditions we used in the derivation. Consider ξ -2 1 = ξ 2 = ξ 3 = √ 2 c log c 1 + c 2 log c 2 c 2 . Then, the last term in the above is c exp {-1 c c 2 2 ( ξ 2 ∧ ξ 3 ) 2 +2log c 1 } = c √ c 2 . In addition, the requirement that ξ 2 → 0 will follow from the assumption that log( c 1 ) / c 2 2 → 0. The other terms in the error bound can be shown to be bounded by c (log 5 ( n c 1 ) / c 2 2 ) 1 4 as well.

Proof of Theorem 5 . The proof again relies on bounding the errors in the normal approximation and variance estimation. Let z be a standard normal random vector of suitable dimension. Then

<!-- formula-not-decoded -->

By the construction of the critical value, cv( α, ϕ ( a | S )), the first term is exactly 1 -α . As a result, the error term in the theorem can be taken as

<!-- formula-not-decoded -->

or any further bound thereof. In the following, we first provide a lemma on normal approximation. Define R ϕ ( a | S ) as the matrix extracting the relevant choice probabilities for constructing the lower bound in the theorem. We use σ ϕ ( a | S ) to collect the standard deviations of R ϕ ( a | S ) ˆ π , and its estimate is represented by ˆ σ ϕ ( a | S ) .

Lemma A.8. The following normal approximation holds

<!-- formula-not-decoded -->

The next step is to replace the infeasible standard errors by its estimate. The following lemma provides an error bound which arises as we 'take the hat off.'

Lemma A.9. Let ξ 1 , ξ 2 &gt; 0 with ξ 2 → 0. Then

<!-- formula-not-decoded -->

To close the proof of the theorem, we provide the further bound that

<!-- formula-not-decoded -->

√

where the second term follows from Lemma A.7. Finally, we set ξ -2 1 = ξ 2 = 2 c log c 1 + c 2 log c 2 c 2 . Proof of Theorem 6 . We first prove necessity. Suppose that the choice is represented by the model. Therefore, there exists τ ∗ such that π ( a, S ) = τ ∗ ( { (Γ , ≻ ) : a is ≻ -best in Γ( S ) } ). Note that the type space AO×P is finite. Therefore, we could write instead π ( a, S ) = ∑ m j =1 τ ∗ ( { (Γ j , ≻ j ) } ) ✶ { a is ≻ -best in Γ( S ) } for some enumeration of the type space { (Γ j , ≻ j ) } m j =1 . Then, given a finite sequence { ( a i , S i ) } n i =1 , we have

<!-- formula-not-decoded -->

Note that the first inequality comes from the fact that τ ∗ is an element in the compact and convex set ∆( AO × P ), which is the set of all probability distribution over the type space. The second equality results from maximizing a linear function over the space ∆( AO × P ), which is a

compact and convex space, and the function achieves its maximum at the extreme point that gives the highest value in the function ∑ n i =1 ✶ { a i is ≻ -best in Γ( S i ) } , and allocates the unit mass on some (Γ , ≻ ).

For sufficiency of the proof, we utilize the proof technique in Chambers and Echenique (2016). We use the same enumeration of the type space as before { (Γ j , ≻ j ) } m j =1 . For simplicity, we let d j ( a, S ) := ✶ { a is ≻ j -best in Γ j ( S ) } . We enumerate all choice problem ( a, S ) ∈ D , so that we have { ( a k , S k ) } p k =1 . Then, there exists a representation for the choice data if there exists a solution, τ : { 1 , ..., m } → [0 , 1] such that τ · (1 , .., 1) = 1, to the following system of linear equation,

<!-- formula-not-decoded -->

We will denote the above system by D τ = π , and also denote d j := ( d j ( a 1 , S 1 ) , d j ( a 2 , S 2 ) , ..., d j ( a p , S p )) for j = 1 , ..., m . We will use the following version of Farkas' Lemma (see, e.g., Chambers and Echenique, 2016).

Lemma A.10 (Farkas' Lemma) . The following two statements are equivalent:

- 1) There is no solution to the system D τ = π where τ · (1 , ..., 1) = 1 and τ ≥ 0,
- 2) There is a vector η and a scalar θ such that

<!-- formula-not-decoded -->

We will demonstrate that if the π satisfies (5), then there is no solution to the Equ (6). We proceed by contradiction. First, we suppose that there exists a solution to (6), η and θ , and the entries of η are non-negative integers. Since η are non-negative integers, there must be at least one entry of η is positive. For entries of η that is a positive number (integer), we set ( b ( k,t ) , B ( k,t ) ) := ( a k , S k ) for all k = 1 , ..., p for all t = 1 , ..., η ( k ), and we contruct a sequence (by repeating ( a k , S k ) for η ( k )th time)

<!-- formula-not-decoded -->

where we omit the element ( b ( k,t ) , B ( k,t ) ) for all t if η ( k ) = 0. By the definition of the sequence, we have ∑ p k =1 ∑ η ( k ) t =1 π ( b ( k,t ) , B ( k,t ) ) = η · π . Similiarly, for any j = 1 , ..., m , we have

∑ p k =1 ∑ η ( k ) t =1 d j ( b ( k,t ) , B ( k,t ) ) = η · d j . Since η and θ are the solution to the system (6), we have

<!-- formula-not-decoded -->

By cancelling out θ on both side, the above is a direct violation of (5). Contradiction arises.

̸

Finally, we show if there exists a solution ( η, θ ) to (6), then there exists η ′ whose entries are non-negative integers such that for j = 1 , .., m , η ′ · π &gt; η ′ · d j holds, which is sufficient to deliver the above contradiction (by choosing the corresponding sequence). Suppose ( η, θ ) is the solution to (6), then it is immediate that η · π &gt; η · d j by using (6) and cancelling out θ . One can first η to have rational entries and satisfy the condition. Then, if we multiply both side with a large enough positive integer, we can take η to have integer entries and satisfy the condition. Therefore, without loss of generality, we assume η is integer-valued. Lastly, we show that whenever there exist k ∈ { 1 , ..., p } such that η ( k ) &lt; 0, then η ′ π &gt; π d j holds for some η ′ such that η ′ ( k ) = 0 and η ≤ η ′ . Suppose that η ( k ) &lt; 0. Then, we define η ′ ( k ) := 0, η ′ ( l ) := η ( l ) -η ( k ) if S l = S k and l = k , and η ′ ( l ) := η ( l ) otherwise. Then, for j = 1 , ..., m , we have η ′ · π &gt; η ′ · d j iff

̸

̸

<!-- formula-not-decoded -->

̸

̸

iff η · π &gt; η · d j , where the first and second iff is given by substituting η ′ ( l ) and the fact that ∑ l : S l = S k ρ ( a l , S l ) = 1 and ∑ l : S l = S k d j ( a l , S l ) = 1 for j = 1 , ..., m . Since p is a finite number, one can perform this procedure in finite steps and reaching η ′ whose entries are all non-negative integers. Therefore, the proof is complete.

̸

Proof of Theorem 7 . Given a linear order ▷ , we consider a partition of P ▷ . We denote P ▷ ( x ) as the subset of single improvement of ▷ where each ≻ in P ▷ ( x ) captures each improvement of x in ▷ . Also, ⋃ x ∈ X P ▷ ( x ) = P ▷ and P ▷ ( x ) ∩ P ▷ ( y ) = ∅ for x = y . For each type (Γ , ≻ ), we denote c (Γ , ≻ ) ( . ) as the corresponding choice function. Multiple types might exhibit the same choice behavior. Moreover, we also define induced choice data given τ . To do that, we first define π τ z given each z where π τ z ( x, S ) := τ ( { (Γ , ≻ ) ∈ AO ▷ × P ▷ ( z ) : c (Γ , ≻ ) ( S ) = x } . It is easy to see that ∑ z ∈ X π τ z ( x, S ) = π τ ( x, S ). We first prove the necessity. To proceed, we denote m S,▷ as the top element in S according to ▷ . Also, we denote m := m X,▷ . For Axiom 1, we proceed by proving two claims. The first one shows that each type chooses either the first item in the decision problem or their reference points.

Claim 1. For any (Γ , ≻ ) ∈ CF ▷ × P ▷ ( x ) and all S , we must have either c (Γ , ≻ ) ( S ) = m S,▷ or c (Γ , ≻ ) ( S ) = x . Therefore, if x / ∈ S , we must have c (Γ , ≻ ) ( S ) = m S,▷ .

̸

Proof. Suppose not. i.e. c (Γ , ≻ ) ( S ) = y and y / ∈ { m S,▷ , x } . Since y = m S,▷ , we must have m S,▷ ▷ y . Since each linear order ≻ in P ▷ ( x ) agrees with ▷ except for x , we have m S,▷ ≻ y . However, by the definition of choice function and the property of competition filter on list, it must be that y ≻ m S,▷ , which implies that y ≻ m S,▷ ≻ y , a contradiction.

Then the second claim is that when x is not the first item in the list, then π τ z ( x, · ) decreasing in the second component when z = x , and is constant otherwise.

̸

Claim 2. For all S ⊆ T and x, z ∈ S with x = z such that there exists y ▷ x and y ∈ S , we have ( i ) π τ x ( x, S ) ≥ π τ x ( x, T ); ( ii ) π τ z ( x, S ) = π τ z ( x, T ) = 0.

̸

̸

Proof. Since there exists y ▷ x and y ∈ S ⊆ T , we know that x = m S,▷ and x = m T,▷ . Then, by Claim 1, for z = x , it must be that c (Γ , ≻ ) ( S ) = x for any (Γ , ≻ ) ∈ CF ▷ ×P ▷ ( z ) and for any S ⊆ X . Therefore, π τ z ( x, S ) = π τ z ( x, T ) = 0. ( ii ) is proven.

̸

̸

̸

For ( i ), it suffices to show that for each ≻∈ P ▷ ( x ) and any Γ, if c (Γ , ≻ ) ( T ) = x , then it must be that c (Γ , ≻ ) ( S ) = x . To see this, suppose not. Then, by Claim 1, we must have c (Γ , ≻ ) ( S ) = m S,▷ . However, since c (Γ , ≻ ) ( T ) = x , we know that it must be that x ≻ m T,▷ . Since x ∈ Γ( T ), by competition filter, we must have x ∈ Γ( S ) so that m S,▷ ≻ x . If m T,▷ = m S,▷ , then x ≻ m T,▷ = m S,▷ ≻ x , which is a contradiction. On the other hand, if m T,▷ = m S,▷ , then it must be that m T,▷ ▷ m S,▷ . Since ≻∈ P ▷ ( x ), it agrees with ▷ over binary relation on X \ x . We have m T,▷ ≻ m S,▷ . However, we can then deduce that x ≻ m T,▷ ≻ m S,▷ ≻ x , a contradiction.

̸

Using the above claim, we can see that for all S ⊆ T and x ∈ S such that there exists y ▷ x and y ∈ S , we have π ( x, S ) = ∑ z ∈ X π τ z ( x, S ) ≥ ∑ z ∈ X π τ z ( x, T ) = π τ ( x, T ). Axiom 1 is proven. For Axiom 2, due to Claim 2, it suffices to show that π τ x ( x, { x, y } ) ≥ π τ x ( x, { x, z } ) for all x, y, z such that z▷y▷x . Since Γ has full attention over binary sets, for any (Γ , ≻ ) ∈ CF ▷ ×P ▷ ( x ), if c (Γ , ≻ ) ( { x, z } ) = x , then it must be that c (Γ , ≻ ) ( { x, y } ) = x . Note that c (Γ , ≻ ) ( { x, z } ) = x implies that x ≻ z . Also, since ≻ agrees with ▷ over binary relation on X \ x , we know that z ≻ y . It must be that x ≻ y . Due to full attention over binary sets, it must be that c (Γ , ≻ ) ( { x, y } ) = x . Hence, it is proven. For Axiom 3, it suffices to show that for x = m , we have τ ( { (Γ , ≻ ) ∈ AO ▷ ×P ▷ ( x ) } ) = π τ ( x, { b x , x } ). It is because if it holds, then 1 -∑ x ∈ X \ m π τ ( x, { b x , x } ) = τ ( { (Γ , ≻ ) ∈ AO ▷ ×P ▷ ( m ) } ) ≥ 0. To see why it holds, note that for any (Γ , ≻ ) ∈ CF ▷ ×P ▷ ( x ), it must be that c (Γ , ≻ ) ( { b x , x } ) = x , since ≻ is a single improvement of ▷ and Γ has full attention over binary sets. Necessity is complete.

We then prove sufficiency. We put the focus on the choice functions generated from (Γ , ≻ ) ∈ CF ▷ ×P ▷ ( x ). We first state a characterization of this type of choice function

̸

Claim 3. Let x ∈ X . A choice function c is represented by some (Γ , ≻ ) ∈ CF ▷ ×P ▷ ( x ) if and only if i) whenever c ( S ) = m S,▷ , we have c ( S ) = x ; ii) c ( T ) = x implies c ( S ) = x for x ∈ S ⊆ T ; iii) c ( { y, x } ) = x for some y ∈ X implies c ( { z, x } ) = x for all z ◁ y ; iv) c ( { z, x } ) = x for all z ◁ x .

Proof. We first prove the only-if part. Note that i ) is given by Claim 1. For ii ), suppose not. Then, by ( i ), it must be that c ( S ) = m S,▷ ▷ x . By competition filter on list, we know that i) m S,▷ , x ∈ Γ( S ) ⊆ Γ( T ). However, c ( S ) implies x ▷ m S,▷ but c ( S ) = m S,▷ implies m S,▷ ▷ x . A contradiction. For iii ), since Γ assign full attention over binary sets, if c ( { y, x } ) = x , by ≻∈ P ▷ ( x ), we have x ≻ z for all z ◁ y . For iv ), it is clear that Γ( { z, x } ) = { z, x } and x ≻ z as x ▷ z . Only-if part is complete.

̸

For the if-part, we can construct (Γ c , ≻ c ) as follows: Firstly, we consider Γ c . For | S | &gt; 2, We let Γ c ( S ) = { m S,▷ } for every S such that c ( S ) = m S,▷ and we let Γ c ( S ) = U ▷ ( x ) ∩ S for every S such that c ( S ) = x = m S,▷ . For | S | = 2, we let Γ c ( S ) = S . Secondly, for ≻ c , we construct the binary relation as follows. For y, z = x , we set y ≻ c z if y▷z for every y, z = x . For binary relations involving x , we set x ≻ c y if c ( { x, y } ) = x , y ≻ c x if c ( { x, y } ) = y .

̸

̸

̸

To check that (Γ c , ≻ c ) ∈ CF ▷ ×P ▷ ( x ), we first check preference. To check that it is complete, for y, z = x , it follows from ▷ . For x, y , we must have either x ≻ c y or y ≻ c x since c ( { x, y } ) is non-empty. To show that it is transitive. For w,y, z = x , if w ≻ y and y ≻ z , we must have w ≻ z since it follows from ▷ . Otherwise, firstly, consider that w ≻ c x and x ≻ c z . Therefore, we know that c ( { w,x } ) = w and c ( { x, z } ) = x . Hence, we know that w ▷ x and x ▷ z . Hence, it must be that w ▷ z . Hence, by i ) it must be that c ( { w,z } ) = w . Secondly, consider that x ≻ c w and w ≻ c z . Then, we know that c ( { w,x } ) = x and w ▷ z . Then, by iii ), we know that c ( { z, x } ) = x . Lastly, consider that w ≻ c z and z ≻ c x , then we know w ▷ z and c ( { z, x } ) = z . It must be that c ( { w,x } ) = w . Suppose not, i.e. c ( { w,x } ) = x . Then, by iii ), we must have c ( { z, x } ) = x . A contradiction. Transitivity is complete. By construction, it is clear that ≻ c ∈ P ▷ ( x ).

̸

̸

Then, we check the consideration set mapping Γ c . Firstly, it satisfies full attention over binary set by constructions. Secondly, to see that it is a competition filter on list ▷ , we let S ⊆ T and y ∈ Γ c ( T ). Let z ∈ U ▷ ( y ) ∩ S , we aim to show z ∈ Γ c ( S ). By construction, it must be either Γ c ( T ) = m T,▷ or Γ c ( T ) = U ▷ ( x ) ∩ T . If y = m T,▷ , then it must be that U ▷ ( y ) ∩ S = { y } and z = y . We then have z ∈ Γ c ( S ) since it is the top element on the list. If y = m T,▷ , then it must be that Γ c ( T ) = U ▷ ( x ) ∩ T so that c ( T ) = x and y ▷ x . By ii ), we have c ( S ) = x so that Γ c ( S ) = U ▷ ( x ) ∩ S . Since z ∈ U ▷ ( y ) ∩ S , z ▷ y ▷ x , we have z ∈ Γ c ( S ).

̸

Lastly, we show that it explains the choice function. For | S | = 2, it is immediate; for | S | &gt; 2, notice that c (Γ c , ≻ c ) ( S ) = m ▷,S if and only if Γ( S ) = { m ▷,S } if and only if c ( S ) = m ▷,S . If c ( S ) = x = m ▷,S , we have Γ c ( S ) = U ▷ ( x ) ∩ S . By ii ), we must have c ( { x, z } ) = x for every z ∈ S . Hence, c (Γ c , ≻ c ) ( S ) = x . The proof is complete.

̸

The idea of the rest of the proof goes as follows. We denote the set of all choice functions generated from (Γ , ≻ ) ∈ CF ▷ × P ▷ ( x ) as ❈ ▷,x . We will construct a sequence of choice functions { c 1 , c 2 , ..., c n } where c 1 ( S ) = x for all S containing x and whenever c i ( S ) = x , we have c i ( S ) = m S,▷ for all i . We denote the set of all such sequences as ◗ ▷,x with typical element q x . Then, given the choice rule π , for each x = m , we select a sequence q x ∈ ◗ ▷,x . We will show that the q x that we choose is a subset of ❈ ▷,x . We will then assign weights to them and show that they jointly explain the data. To abuse notation, we write τ ( . ) as also the probability measure over choice functions ⋃ x ∈ X ❈ ▷,x . Firstly, for each x ∈ X and x = m , we gather all the conditions from Axiom 2 and 3 which are related to x , which are π ( x, S ) ≥ π ( x, T ) for x = m S,▷ and S ⊆ T ; π ( x, { x, y } ) ≥ π ( x, { x, z } ) for z ▷ y ▷ x .

̸

̸

We denote the sets containing x and appearing in the inequalities as S x . We let B 0 ⊆ S x be the sets in S x that is non-dominating : If there does not exist S ′ ∈ S x and S ′ = S such that π ( x, S ) ≥ π ( x, S ′ ), then S ∈ B 0 . Therefore, B 0 = { X } . Then, we construct B i for i = 1 , ..., |S x | and B i for i = 1 , ..., |S x | as follows:

<!-- formula-not-decoded -->

We assume that the minimizer B i is unique for simplicity. Hence, since B 0 = { X } , we have B 1 = X . Also, B 1 equals the sets of size N -1 containing x . Lastly, since minimizer is unique and there are |S x | sets to begin with, we have B |S x | = { b x , x } , where b x is the immediate predecessor of x in X according to ▷ , as p ( x, { b x , x } ) ≥ p ( x, S ) for all S ∈ S x by Axiom 2 and Axiom 3. Then, we pick q x = { c 1 , c 2 , .., c |S x | } ∈ ◗ ▷,x such that, for i = 2 , 3 , ... |S x | , c i ( S ) = x if S / ∈ { B 1 , ..., B i -1 } and c i ( S ) = m S,▷ if S ∈ { B 1 , ..., B i -1 } .

We first verify that each of these choice functions c i ∈ q x belongs to ❈ ▷,x . We check each condition in Claim 3. For i ) it is immediate. For ii ), assume that c i ( T ) = x and let S ⊂ T . Since c i ( T ) = x , it must be that T / ∈ { B 1 , ..., B i -1 } . Then, it must be that S / ∈ { B 1 , ..., B i -1 } . To see this, suppose not, i.e. S ∈ { B 1 , ..., B i -1 } and let S = B j and j ≤ i -1. Since S = B j := arg min S ′ ∈B j -1 ρ ( x, S ), we know S ∈ B j -1 . B j -1 are the sets in S x \ { B 0 , ..., B j -1 } that are non-dominating. Yet, since T / ∈ { B 1 , ..., B i -1 } , we have T ∈ S x \ { B 0 , ..., B j } . Therefore, S is dominating in S x \ { B 0 , ..., B j } since ρ ( x, S ) ≥ ρ ( x, T ) by Axiom 2. A contradiction. For iii ),

̸

̸

let c i ( { y, x } ) = x , we need to show c i ( { z, x } ) = x for all z ▷ y . One can prove it using the same argument as above by using Axiom 3. For iv ), for all z ▷ x , the sets { z, x } do not appear in Axiom 2 or 3. Hence, { z, x } / ∈ S x and c i ( { z, x } ) = x for all i .

̸

Then, for x = m , we assign weights to q x . In particular, we let τ π ( { c 1 } ) = π ( x, B 1 ) ≥ 0 and τ π ( { c i } ) = π ( x, B i ) -π ( x, B i -1 ) ≥ 0 for i = 2 , ..., |S x | . These are non-negative by construction. Also, It is easy to see that τ π ( q x ) = π ( x, { b x , x } ) since B |S x | = { b x , x } . Lastly, we endow the (unique) choice function c ∗ in ❈ ▷,m with weight 1 -∑ x ∈ X \ m π ( x, { b x , x } ), i.e. τ π = 1 -∑ x ∈ X \ m π ( x, { b x , x } ), which is non-negative by Axiom 4. Hence, we get τ π ( ⋃ x ∈ X ❈ ▷,x ) = ∑ x ∈ X τ π ( ❈ ▷,x ) = τ π ( { c ∗ } ) + ∑ x = m τ π ( q x ) = 1 -∑ x ∈ X \ m π ( x, { b x , x } ) + ∑ x ∈ X \ m π ( x, { b x , x } ) = 1. Hence, τ π is a probability measure. Moreover, to see that τ π explains the choice data, consider x = m S,▷ , for every S , π τ π ( x, S ) = τ π ( { c ∈ ❈ ▷,x : c ( S ) = x } ) = τ π ( { c 1 , c 2 , ..., c n : B n = S } ) = π ( x, B 1 ) + ∑ n i =2 π ( x, B i ) -π ( x, B i -1 ) = π ( x, B n ) = π ( x, S ). Lastly, for every S , π τ π ( m S,▷ , S ) = 1 -∑ x = m S,▷ π τ π ( x, S ) = 1 -∑ x = m S,▷ π ( x, S ) = π ( m S,▷ , S ). Hence, it explains the choice rule. On the other hand, when the minimizers B i 's are not unique, one can set assign zero weight for some choice functions in q x and the proof is basically the same. Hence, the proof is complete.

̸

̸

Proof of Theorem 8. Since Γ assigns full attention at S when | S | = 2, we focus on the preference types and skip Γ when denoting the choice function for each preference type for menus of two alternatives. Consider choice probability at S = { a i , a j } and i &gt; j . Notice that for types ≻ kl where k = j , c ≻ kl ( { a i , a j } ) = a j , since the list ▷ agree with ≻ kl over alternatives other than a k ; for types ≻ kl where k = j and i &lt; l , c ≻ kl ( { a i , a j } ) = a j , since ≻ kl still rank a i higher than a j even though a j is moved higher; lastly, for types ≻ kl where k = j and i ≥ l , c ≻ kl ( { a i , a j } ) = a i , since ≻ kl moves a i higher than a k . Therefore, we know that, for i &gt; j π ( a i , { a i , a j } ) = τ { (Γ , ≻ il ) : l ≤ j } . Hence, for i = j = 1, we have π ( a i , { a i , a j } ) -π ( a i , { a i , a j -1 } ) = τ { (Γ , ≻ il ) : l ≤ j } - { (Γ , ≻ il ) : l ≤ j -1 } = τ ( ≻ ij ). Also, for i = j = 1, π ( a i , { a i , a j } ) = τ { (Γ , ≻ il ) : l ≤ 1 } = τ ( ≻ i 1 ). Lastly, given a linear order ▷ , we consider a partition of P ▷ . We denote P ▷ ( x ) as the subset of single improvement of ▷ where each ≻ in P ▷ ( x ) only disagrees with ▷ over x . By definition, τ ( P ▷ ( a k )) = τ { (Γ , ≻ kl ) : l ≤ k -1 } = π ( a k , { a k -1 , a k } ) for k = 1. Hence, for ▷ , which is denoted as ≻ ii by an abuse of notation, we have τ ( ▷ ) = 1 -∑ n k =2 τ ( P ▷ ( a k )) = 1 -∑ n k =2 π ( a k , { a k -1 , a k } ). Proof of Theorem 9 . We first show the lower bound. Suppose that the maximum is achieved at R ⊇ S . We enumerate the alternative in R by ⟨ a r 1 ,. . . , a r | R | ⟩ so that a k = a r ℓ and 1 &lt; ℓ . For some a r i where i ≥ ℓ to be chosen, it must be that a r i is considered by certain choice types. Therefore, by List-based Attention Overload, in set R , these choice types must also have considered everything before a r i , including a r ℓ . Hence, we have ϕ ( a k | R ) ≥ ∑ ℓ ≥ k π ( a ℓ | R ). Lastly, since Listbased Attention Overload satisfies Attention Overload, it must be that ϕ ( a k | S ) ≥ ϕ ( a k | R ).

̸

̸

̸

̸

̸

̸

̸

In the following, we let U ▷ ( a k ) be the alternatives in X which are listed before a k (including a k ); that is, it is the weak upper contour set of a k according to the list order. Therefore, for the upper bound, it suffices to show the bound ϕ ( a k | S ) ≤ 1 -∑ b ∈ U ▷ ( a k ) ∩ S \ a s 1 ( π ( b | a s 1 ) -π ( b | S )). It is because, given by Axiom 2, max R ⊇{ a s 1 ,b } π ( b | R ) = π ( b | a s 1 ) and min a s 1 ,b ∈ T ⊆ S π ( b | T ) = π ( b | S ). Firstly, fix an b ∈ U ▷ ( a k ) ∩ S \ a s 1 . For b to be chosen is S , it must have been considered by the preference types which rank b before a s 1 . i.e. (Γ , ≻ ) where ≻∈ {≻ ba s 1 , ≻ ba s 1 -1 , ..., ≻ ba 1 } . If all of these types have paid attention to b in S , then π ( b | a 1 ) -π ( b | S ) = 0, since full attention is assumed at binary sets. Therefore, the difference π ( b | a 1 ) -π ( b | S ) captures the types which have not noticed b but would have chosen b if otherwise it is (counterfactually) discovered. Also, by List-based Attention Overload, these types must not have considered a k , since a k is after b . Since the types are independent, i.e. for two different b, b ′ , (Γ , ≻ ) where ≻∈ {≻ ba s 1 , ≻ ba s 1 -1 , ..., ≻ ba 1 } and (Γ , ≻ ) where ≻∈ {≻ b ′ a s 1 , ≻ b ′ a s 1 -1 , ..., ≻ b ′ a 1 } are independent, ∑ b ∈ U ▷ ( a k ) ∩ S \ a s 1 ( π ( b | a s 1 ) -π ( b | S )) reveals the types who must not have paid attention to a k . Therefore, 1 -∑ b ∈ U ▷ ( a k ) ∩ S \ a s 1 ( π ( b | a s 1 ) -π ( b | S )) is an upper bound for ϕ ( a k | S ).

Lastly, we show that the upper bound is greater than the lower bound. Suppose that the maximum for the lower bound is achieved at R ⊇ S , then 1 -∑ b ∈ U ▷ ( a k ) ∩ S \ a s 1 ( π ( b | a s 1 ) -π ( b | S )) -∑ j ≥ k π ( a j | R ) ≥ (1 -∑ a s j ∈ U ▷ ( a k ) ∩ S \ a s 1 π ( a s j | a s 1 )) -∑ j&gt;k π ( a j | R ) ≥ (1 -∑ a s j ∈ U ▷ ( a k ) ∩ S \ a s 1 π ( a s j | a s j -1 )) -∑ j&gt;k π ( a j | R ) ≥ ∑ j&gt;k π ( a j | a j -1 ) -∑ j&gt;k π ( a j | R ) ≥ ∑ j&gt;k π ( a j | a s 1 ) -∑ j&gt;k π ( a j | R ) ≥ 0. The second, third, fourth, and fifth inequalities are given by Axiom 3, Axiom 4, Axiom 3, and Axiom 2, respectively. Therefore, the proof is complete.

## B Appendix: Unknown List

We relax the assumption that the list ▷ is known, and show how to obtain preference elicitation in our choice model. We describe to what extent one can identify the list of a given probabilistic choice function for a HAOM ▷ representation. First, we introduce the following definition.

̸

Definition 5 (Strict Choice Rule) . A probabilistic choice function π is strict if π ( a | b ) = π ( a | c ) for all distinct a, b, c .

For the revelation of the list, we assume that π is strict and that the model HAOM ▷ is correctly specified, and we investigate whether the underlying list is identifiable from choice data. There are at least two ways to elicit ▷ in our choice model. First, if we have π ( a | S ) &lt; π ( a | T ) for some a ∈ S ⊆ T , then we know that a must appear before all alternatives in T by Axiom 2. Second, we can use the information coming from binary menus. Suppose we have π ( c | b ) &gt; π ( c | a ), we can

immediately conclude that it cannot be b ▷ a ▷ c by Axiom 3. In other words, if c is after a and a is after b , c should be chosen more often with the one ranked closer. Since we observe the opposite situation, then it must not be that c is after a , and a is after b . Note that π ( c | b ) &gt; π ( c | a ) also implies π ( c | b )+ π ( a | c ) &gt; 1, which would violate the boundedness imposed by Axiom 4 if it is b▷c▷a . On the other hand, by analogy, observing π ( b | c ) &gt; π ( b | a ) will rule out the other two possibilities, i.e. c ▷ a ▷ b and c ▷ b ▷ a . Therefore, the remaining possibilities are a ▷ b ▷ c and a ▷ c ▷ b . In either case, we know that a appears before b and c . Let a L π b if (i) there exists { a, b } ⊆ S ⊆ T such that π ( a | S ) &lt; π ( a | T ), or (ii) there exists c such that π ( c | b ) &gt; π ( c | a ) and π ( b | c ) &gt; π ( b | a ) .

Our discussion shows that if any π has a HAOM ▷ representation, then L π must be a subset of ▷ . While this is an important observation, the application could be limited if L π is incomplete. The next theorem illustrates that L π is 'almost' complete. That is, L π includes all binary comparisons except the binary comparison of the last two alternatives in the list. In other words, L π identifies the list up to the last two elements. It is possible that some data also reveals the position of the last two elements.

Theorem 12. If a strict π has a HAOM ▷ representation, the list is uniquely identified up to the last two elements by L π .

Proof of Theorem 12. Suppose a strict π has a HAOM ▷ representation in ▷ where a 1 ▷ a 2 ... ▷ a n . We will show that it must be that L π = ▷ \ { ( a n -1 , a n ) } .

We will first prove ⊆ . Suppose there exists { x, y } ⊆ S ⊆ T such that π ( x, S ) &lt; π ( x, T ). By Axiom 2, it must be that x is the ▷ -most in S and hence x▷y . On other hand, suppose there exist z ∈ X such that π ( z, { y, z } ) &gt; π ( z, { x, z } ) and π ( y, { y, z } ) &gt; π ( y, { x, y } ). By π ( z, { y, z } ) &gt; π ( z, { x, z } ) and Axiom 3, we know that it must not be the case that y ▷ x ▷ z . Also, by a rearrangement, π ( z, { y, z } ) &gt; π ( z, { x, z } ) implies π ( z, { y, z } ) + π ( x, { x, z } ) &gt; 1. It must not be the case that y ▷ z ▷ x . To see this, suppose y ▷ z ▷ x . Suppose the immediate predecessor of z in X is z X , and the immediate predecessor of x in X is x X . Axiom 4 implies that ρ ( z, { z X , z } ) + ρ ( x, { x, x X } ) ≤ 1. Also, by Axiom 4, we know that ρ ( z, { z X , z } ) ≥ ρ ( z, { y, z } ) and ρ ( x, { x, x X } ) ≥ ρ ( x, { x, z } ). Therefore, we have ρ ( z, { y, z } ) + ρ ( x, { x, z } ) ≤ 1. A contradiction. Therefore, it cannot be y ▷ z ▷ x . Analogously, π ( y, { y, z } ) &gt; π ( y, { x, y } ) imply that it cannot be either z▷x▷y or z▷y▷x . Therefore, it must be either x ▷ y ▷ z or x ▷ z ▷ y . In either case, we have ( x, y ) ∈ ▷ .

̸

For ⊇ , suppose ( a k , a l ) ∈ ▷ where k &lt; l and ( a k , a l ) = ( a n -1 , a n ). Therefore, there exists a h such that k &lt; l &lt; h . Also, since π is strict, by Axiom 3, it must be that π ( a h , { a h , a l } ) &gt; π ( a h , { a h , a k } ). On the other hand, it must also be that π ( a l , { a l , a h } ) &gt; π ( a l , { a l , a k } ). Suppose instead π ( a l , { a l , a h } ) &lt; π ( a l , { a l , a k } ). Therefore, we have 1 &lt; π ( a l , { a l , a k } ) + π ( a h , { a l , a h } ).

Axiom 4 implies that ρ ( a l , { a l , a l -1 } ) + ρ ( a h , { a h , a h -1 } ) ≤ 1. Then, by Axiom 3, we have π ( a l , { a l , a k } ) + π ( a h , { a l , a h } ) ≤ 1. A contradiction. Therefore, we have both π ( a h , { a h , a l } ) &gt; π ( a h , { a h , a k } ) and π ( a l , { a l , a h } ) &gt; π ( a l , { a l , a k } ). Then, we have ( a k , a l ) ∈ L π . The proof is done.

Theorem 12 guarantees that the list is almost point identified: L π is missing one binary comparison. Hence, there are two completions of L π . It is routine to check whether Axioms 2-4 are satisfied by (at least) one of those completions. We can state the following corollary as the characterization result for the unknown list environment.

Corollary 3 (Characterization) . A strict π has a HAOM ▷ representation if and only if (i) L π ranks everything except the last two alternatives and (ii) π satisfies Axioms 2-4 according to one possible completion of L π .

For necessity, the first requirement is given by Theorem 12, and the second one is given by the fact that π has a HAOM ▷ representation in ▷ , which is a completion of L π according to Theorem 12. The sufficiency is given by Theorem 7.

## References

- Abaluck, J., and A. Adams (2021): 'What Do Consumers Consider Before They Choose? Identification from Asymmetric Demand Responses,' Quarterly Journal of Economics , 136(3), 1611-1663.
- Aguiar, V. H. (2017): 'Random Categorization and Bounded Rationality,' Economics Letters , 159, 46-52.
- Aguiar, V. H., M. J. Boccardi, and M. Dean (2016): 'Satisficing and stochastic choice,' Journal of Economic Theory , 166, 445-482.
- Augenblick, N., and S. Nicholson (2016): 'Ballot position, choice fatigue, and voter behaviour,' Review of Economic Studies , 83(2), 460-480.
- Barseghyan, L., M. Coughlin, F. Molinari, and J. C. Teitelbaum (2021): 'Heterogeneous Choice Sets and Preferences,' Econometrica , 89(5), 2015-2048.
- Barseghyan, L., F. Molinari, and M. Thirkettle (2021): 'Discrete Choice under Risk with Limited Consideration,' American Economic Review , 111(6), 1972-2006.
- Biswas, D., D. Grewal, and A. Roggeveen (2010): 'How the order of sampled experiential products affects choice,' Journal of Marketing Research , 47(3), 508-519.
- Brady, R. L., and J. Rehbeck (2016): 'Menu-Dependent Stochastic Feasibility,' Econometrica , 84(3), 1203-1223.

- Cattaneo, M. D., X. Ma, Y. Masatlioglu, and E. Suleymanov (2020): 'A Random Attention Model,' Journal of Political Economy , 128(7), 2796-2836.
- Chambers, C. P., and F. Echenique (2016): Revealed Preference Theory . Cambridge University Press.
- Chernozhukov, V., D. Chetverikov, and K. Kato (2019): 'Inference on Causal and Structural Parameters using Many Moment Inequalities,' Review of Economic Studies , 86(5), 18671900.
- Chernozhukov, V., D. Chetverikov, K. Kato, and Y. Koike (2022): 'Improved Central Limit Theorem and Bootstrap Approximations in High Dimensions,' Annals of Statistics , 50(5), 2562-2586.
- Dardanoni, V., P. Manzini, M. Mariotti, and C. J. Tyson (2020): 'Inferring Cognitive Heterogeneity From Aggregate Choices,' Econometrica , 88(3), 1269-1296.
- Demirkan, Y., and M. Kimya (2020): 'Hazard Rate, Stochastic Choice and Consideration Sets,' Journal of Mathematical Economics , 87, 142-150.
- Ellison, G., and S. F. Ellison (2009): 'Search, obfuscation, and price elasticities on the internet,' Econometrica , 77(2), 427-452.
- Fishburn, P. C. (1998): 'Stochastic Utility,' in Handbook of Utility Theory , ed. by S. Barbera, P. Hammond, and C. Seidl, pp. 273-318. Kluwer Dordrecht.
- Geng, S. (2016): 'Decision Time, Consideration Time, and Status Quo Bias,' Economic Inquiry , 54(1), 433-449.
- Guney, B. (2014): 'A theory of iterative choice in lists,' Journal of Mathematical Economics , 53, 26-32.
- Horan, S. (2010): 'Sequential search and choice from lists,' Working Paper .
- Ishii, Y., M. Kovach, and L. ¨ Ulk¨ u (2021): 'A model of stochastic choice from lists,' Journal of Mathematical Economics , 96, 102509.
- Kitamura, Y., and J. Stoye (2018): 'Nonparametric analysis of random utility models,' Econometrica , 86(6), 1883-1909.
- Koshevoy, G., and E. Savaglio (2023): 'On rational choice from lists of sets,' Journal of Mathematical Economics , 109, 102891.
- Kovach, M., and L. ¨ Ulk¨ u (2020): 'Satisficing with a variable threshold,' Journal of Mathematical Economics , 87, 67-76.
- Levav, J., M. Heitmann, A. Herrmann, and S. S. Iyengar (2010): 'Order in Product Customization Decisions: Evidence from Field Experiments,' Journal of Political Economy , 118(2), 274-299.
- Lleras, J. S., Y. Masatlioglu, D. Nakajima, and E. Y. Ozbay (2017): 'When More Is Less: Limited Consideration,' Journal of Economic Theory , 170, 70-85.

- Manzini, P., and M. Mariotti (2014): 'Stochastic Choice and Consideration Sets,' Econometrica , 82(3), 1153-1176.
- Manzini, P., M. Mariotti, and L. ¨ Ulk¨ u (2024): 'A model of approval with an application to list design,' Journal of Economic Theory , 217, 105821.
- Marschak, J. (1959): 'Binary choice constraints and random utility indicators,' in Theory and Decision Library , p. 218-239. Springer.
- Matzkin, R. L. (2007): 'Nonparametric Identification,' in Handbook of Econometrics, Volume VIB , ed. by J. Heckman, and E. Leamer, pp. 5307-5368. Elsevier Science B.V.
- (2013): 'Nonparametric Identification in Structural Economic Models,' Annual Review of Economics , 5, 457-486.
- McFadden, D., and M. K. Richter (1990): 'Stochastic rationality and revealed stochastic preference,' in Preferences, Uncertainty, and Optimality, Essays in Honor of Leo Hurwicz , pp. 161-186. Westview Press: Boulder, CO.
- Molinari, F. (2020): 'Microeconometrics with Partial Identification,' in Handbook of Econometrics, Volume VIIA , ed. by S. Durlauf, L. Hansen, J. Heckman, and R. Matzkin, pp. 355-486. Elsevier Science B.V.
- Pernice, K., K. Whitenton, J. Nielsen, et al. (2018): How People Read Online: The Eyetracking Evidence . Nielsen Norman Group.
- Reutskaja, E., and R. M. Hogarth (2009): 'Satisfaction in Choice as a Function of the Number of Alternatives: When 'Goods Satiate',' Psychology and Marketing , 26(3), 197-203.
- Reutskaja, E., R. Nagel, C. F. Camerer, and A. Rangel (2011): 'Search Dynamics in Consumer Choice under Time Pressure: An Eye-Tracking Study,' American Economic Review , 101(2), 900-926.
- Rubinstein, A., and Y. Salant (2006): 'A model of choice from lists,' Theoretical Economics , 1(1), 3-17.
- Simon, H. A. (1955): 'A behavioral model of rational choice,' Quarterly Journal of Economics , 69(1), 99-118.
- Tserenjigmid, G. (2021): 'The order-dependent Luce model,' Management Science , 67(11), 6915-6933.
- Turansick, C. (2022): 'Identification in the Random Utility Model,' Journal of Economic Theory , p. 105489.
- Tversky, A. (1972): 'Elimination by aspects: A theory of choice,' Psychological review , 79(4), 281.
- Visschers, V. H., R. Hess, and M. Siegrist (2010): 'Health Motivation and Product Design Determine Consumers Visual Attention to Nutrition Information on Food Products,' Public Health Nutrition , 13(7), 1099-1106.

- Westerwick, A. (2013): 'Effects of sponsorship, web site design, and Google ranking on the credibility of online information,' Journal of Computer-Mediated Communication , 18(2), 194211.
- Yegane, E. (2022): 'Stochastic choice with limited memory,' Journal of Economic Theory , 205, 105548.
- Yildiz, K. (2016): 'List-rationalizable choice,' Theoretical Economics , 11(2), 587-599.