## Inferring Cognitive Heterogeneity from Aggregate Choices ∗

Valentino Dardanoni

Universit` a degli Studi di Palermo

Paola Manzini

University of Sussex and IZA

Marco Mariotti

Queen Mary University of London

Christopher J. Tyson

Queen Mary University of London

March 23, 2019

## Abstract

Theoretical work on bounded rationality typically assumes a rich dataset of choices from many overlapping menus, limiting its practical applicability. In contrast, we study the problem of identifying the distribution of cognitive characteristics in a population of agents from a minimal dataset consisting of aggregate choice shares from a single menu. Under homogeneous preferences, we show that both 'consideration capacity' and 'consideration probability' distributions can be recovered effectively when the menu is sufficiently large. This remains generically true when tastes are heterogeneous but their distribution is known. When the preference distribution is unknown, we demonstrate that joint choice share data from as few as three 'occasions' are generically sufficient for full identification of the cognitive capacity distribution and provide substantial information about tastes.

Keywords: attention, bounded rationality, consideration set, stochastic choice.

J.E.L. codes:

D01, D12, D91.

∗ For valuable comments and suggestions that have improved this paper, we would like to thank the co-editor and three anonymous referees; Jason Abaluck, Abi Adams, Levon Barseghyan, Aluma Dembo, Reyer Gerlagh, Alessandro Iaria, Rod McCrorie, Yusufcan Masatlioglu, Irina Merkurieva, Francesca Molinari, and Ted O'Donoghue; seminar audiences at Edinburgh, Leuven, Naples, Oxford, St Andrews, Sussex, Tilburg, ULB (ECARES), Vienna, and Warwick (CRETA); and audiences at the Barcelona GSE Summer Forum (2016), D-TEA (2016), ICEEE (2017), the workshop on 'Identification and Inference in Limited Attention Models' at Cornell University (2018), the Petralia Summer Workshop (2018), and the ASSA meetings (2019). Manzini and Mariotti acknowledge financial support from the ESRC through grant ES/J012513/1, and Dardanoni acknowledges support from MIUR through grant PRIN 2015.

## 1 Introduction

## 1.1 Motivation

Classical revealed preference analysis has yielded a fine-grained understanding of the relationship between unobserved tastes and observed choices, and of how to infer the former from the latter. More recently, theoretical work on bounded rationality has extended this methodology to incorporate a range of cognitive factors that may affect decision making. 1 One drawback of such theories is that they typically presume access to a very rich dataset-comprising a single individual's choices from a large number of different overlapping menus-that can be used to identify the latent components of the cognitive model of interest. For instance, Aguiar et al. [2], Cattaneo et al. [13], and Masatlioglu et al. [32] require data for all possible menus drawn from a universal set of alternatives; Manzini and Mariotti [30] impose a stringent 'richness' assumption on their dataset; and Caplin and Dean [10] postulate the observability of state-dependent stochastic choice data. 2

Identification results developed using such assumptions on the choice domain are often formally elegant, and can be particularly useful for designing and interpreting experiments (as in Aguiar et al. [2] and Caplin and Dean [10]). They are less obviously relevant to field data, however, especially when the type of decision arises rarely (e.g., choice of hospital provider for elective surgery) or the menu is slow to change (e.g., choice of daily newspaper). Indeed, in settings with such features many characterization results from the literature on boundedly rational choice may appear implausibly data-hungry. In practice there may be insufficient menu variation to infer the model components of interest, and for this reason it is desirable to devise approaches to identification that create a direct link between theory and what may be feasible empirically.

1 This literature examines cognitive factors such as computational constraints, norms and heuristics, reference points and other framing effects, and various conceptions of attention. Contributions include those of Apesteguia and Ballester [4], Baigent and Gaertner [6], Caplin et al. [11], Cherepanov et al. [14], de Oliveira et al. [18], Echenique et al. [19], Manzini and Mariotti [29], Masatlioglu and Nakajima [31], Ok et al. [35], Salant and Rubinstein [38], and Tyson [44, 45], among numerous others.

2 Even stronger assumptions about data availability are commonplace in the theory of choice under uncertainty, where the decision maker is typically imagined to express preferences over a highly structured mathematical space specifically designed to facilitate identification.

In this paper we focus on models of limited attention, where agents consider only a subset of the available alternatives, known as the 'consideration set.' 3 To address the data-voracity issue noted above, we propose a novel framework that postulates a minimal dataset comprising (in its basic version) a single, fixed menu from which we observe only the aggregate choice shares of a population of decision makers. 4 Members of the population may (or may not) differ in their preferences over the alternatives, and they may also differ in cognitive characteristics that affect the allocation of attention. The latter 'cognitive heterogeneity' is taken to be unobserved, and our principal goal is to infer the distribution of these characteristics from the aggregate choice shares.

We stress that this paper is concerned with the extent to which the distribution of cognitive characteristics is identified by a given model of bounded rationality per se ; that is, prior to any ancillary econometric specification that may include covariates for the individuals or the alternatives. In this respect our primitives and objectives both remain typical of those in conventional abstract choice theory. Indeed, this is one way that our contribution can be distinguished from other recent work (see, e.g., Abaluck and Adams [1] and Barseghyan et al. [7, 8]) in which identification is facilitated by the presence of observable covariates.

## 1.2 Cognitive models

In our general framework, each agent has a cognitive type parameter q ∈ Q ⊂ glyph[Rfractur] that is distributed in the population according to a cumulative distribution function F . Given preferences over the menu, an individual of type q will choose alternative x with probability p q ( x ) , and the corresponding aggregate choice share will be p ( x ) = ∫ Q p q ( x ) dF . When the cognitive type is used to capture some form of bounded rationality, the indi-

3 This usage follows the marketing literature; see, e.g., Roberts and Lattin [37] and Shocker et al. [39]. While we view the consideration set as a manifestation of bounded rationality, other interpretations are possible. Indeed, alternatives may fail to be considered due to habit formation, search costs, or other forms of rational inattention (see, e.g., Caplin and Dean [10] and Sims [41]).

4 Alternatively, the framework could model a single individual choosing repeatedly from the same menu in different attentional states, where the variation may arise, for example, from a merchandising strategy of the retailer designed to manipulate customers' consideration sets. In Section 4.2 we extend this framework to allow for richer 'multi-occasion' choice data, but only after the informational value of our basic dataset has been completely exhausted.

vidual choice distribution will not generally assign all probability to the best available option, and neither will the aggregate distribution even when the population has homogeneous tastes. Indeed, the very fact that suboptimal alternatives will sometimes be chosen is what will enable us to infer features of the cognitive distribution F from the observed aggregate shares. 5

As mentioned above, we study bounded rationality in the form of limited attention, where the cognitive parameter q influences the formation of the decision maker's consideration set. In the 'consideration capacity' model, the parameter g ∈ { 0, 1, 2, . . . } controls the maximum cardinality of the consideration set and is interpreted as a limit on the number of alternatives that the agent can actively investigate at any one time. We also examine in detail an important special case, the 'consideration probability' model, in which the parameter r ∈ [ 0, 1 ] controls the likelihood that each option is considered and is interpreted as the decision maker's general awareness of the choice environment. We hypothesize that preferences are maximized over the consideration set, and full rationality can be restored by letting g → ¥ or r = 1, as appropriate. 6

## 1.3 Preview of results

We begin by assuming that the population has homogeneous preferences. Under this assumption, we find that our cognitive model is fully identified by a small number of observed choice shares for several natural parameterizations of F . More specifically, if the consideration capacity g has a Poisson or Pascal distribution, or if the consideration probability r has a uniform or Beta distribution, then between one and three aggregate choice shares are needed to recover all of the parameters of the cognitive distribution (see Examples 1-4). We then proceed to show that even in the absence of a parametric

5 Note that our framework has similarities to mixed models in the discrete choice literature, where q would be a taste parameter such as the agent's unobserved marginal utility of some observed characteristic. (See Train [43] and McFadden [33].) However, since we shall use q to control cognition instead of tastes, our setting calls for different functional form assumptions. In particular, p q will not have a logit specification (see Luce [27]), as would typically be assumed in relation to tastes.

6 Variants of the consideration capacity model are employed by Barseghyan et al. [7] to study discrete choice with heterogeneous consideration sets, and by de Clippel et al. [17] to study price competition in a setting where consumers exhibit limited attention. A version of the consideration probability model appears in Manzini and Mariotti [30].

specification, the cognitive distribution can for practical purposes be fully recovered provided the menu of alternatives is sufficiently large. In the context of the consideration capacity model, the choice shares identify the probabilities of all capacities less than the cardinality n of the menu (see Proposition 2). Similarly, in the context of the consideration probability model the choice shares identify the first n raw moments of F (see Proposition 3), which-using maximum entropy methods and results from sparsity theory-can be exploited to reconstruct or to closely approximate the cognitive distribution itself (see Propositions 4-5). In each context, identification follows from the system of equations that define the choice shares being recursive and linear in the relevant quantities (namely, the capacity probabilities or the raw moments), so that closed-form expressions for these quantities can be obtained by inverting a triangular or anti-triangular matrix.

Turning to the case of heterogeneous preferences, we first note that our identification results continue to hold generically if the taste distribution is known (see Propositions 6-7). For heterogeneous and unknown tastes, we extend our dataset to include the joint distribution of choices by the same population of agents on at least three distinct 'occasions'. Making use of an algebraic result on the uniqueness of tensor decompositions, we show that joint choice share data of this sort are generically sufficient for full identification of the cognitive capacity distribution, and also provide substantial information about the distribution of preferences (see Proposition 8).

## 1.4 Related empirical literature

While remaining entirely theoretical, this paper contributes to a growing literature on estimating consideration-set models from consumer demand or other choice data, reviewed briefly in this section.

Abaluck and Adams [1] construct a general econometric framework in which product characteristics are observable, and exploit asymmetries in cross-characteristic choice probability responses to identify consideration sets. Aguiar et al. [2] test random consideration models at the population level in an online experiment, finding support for a specification with heterogeneous preferences and logit attention. In the context

of choice under risk, Barseghyan et al. [7] obtain partial identification of preferences using minimal assumptions about the process of consideration-set formation, while Barseghyan et al. [8] obtain point identification of both preferences and attention in a discrete choice model. Cattaneo et al. [13] postulate 'monotonic attention,' a restriction on how stochastic consideration sets change across menus, and use this assumption to derive testable restrictions on choice probabilities. Crawford et al. [16] devise a model-free identification strategy based on reducing the menu of alternatives to a 'sufficient set' of those that are certain to be considered. Gaynor et al. [20] exploit institutional changes to identify consideration sets in hospital choice, while Honka et al. [23] (among others) treat consideration sets as the outcome of a search process. 7 Lu [26] describes an approach to estimating multinomial choice models that employs known upper and lower bounds on the consideration set. Sovinsky Goeree [42] studies the impact of marketing on the consideration set, using advertising data to separate the utility and attentional components of demand. And Van Nierop et al. [46] propose a model of brand choice accommodating both stated and revealed consideration-set data, which they apply to an experiment on merchandising strategies.

## 1.5 Outline

The remainder of the paper is structured as follows. Section 2 describes our framework and sets out both the consideration capacity model and the special case of the consideration probability model. Section 3 pursues cognitive inference under the simplifying assumption of homogeneous tastes, Section 4 extends the analysis to allow for taste heterogeneity, and Section 5 concludes.

7 The search literature typically deals with datasets that include information about the composition of a consumer's consideration set, although there are exceptions. For example, in Hastings et al. [22] exposure to a sales force influences the probability that financial products are considered.

## 2 Cognitive heterogeneity and consideration-set models

## 2.1 General framework

Let X denote the (finite) universal set of alternatives. A menu is any nonempty A ⊆ X , with which is associated a default outcome dA / ∈ A . When presented with the menu A , an agent either chooses exactly one of the available alternatives or chooses none and accepts dA . For example, we could have that:

- (i) The menu contains retailers selling a product, and the default is not to buy.
- (ii) The menu contains banks offering fixed deposits, and the default is to hold cash.
- (iii) The menu contains risky lotteries, and the default is a risk-free payment.

When deriving our main theoretical results (in Sections 2-3), we shall assume that all agents share the same linear order preferences glyph[followsorequal] over X . This assumption (relaxed in Section 4) can be interpreted as using the average utilities of the alternatives in the population, ignoring individual variation. In this sense our approach is complementary to that of the classical stochastic-choice literature in economics, where preferences are allowed to vary but cognitive capabilities are (implicitly) assumed to be uniform. Note that homogeneous tastes are plausible in examples (i) and (ii) above, where preferences will be determined largely by price and interest rate comparisons, as well as in example (iii) provided all agents are approximately risk neutral over the relevant stakes.

When imposing homogeneous tastes, we number the alternatives so that a higher position in the preference order implies a lower index. We thus write kA for the k th best option on A , and the full menu appears as A = { 1 A , 2 A , . . . , nA } , where nA = | A | .

We introduce cognitive heterogeneity by assigning each agent a cognitive type q ∈ Q ⊂ glyph[Rfractur] , drawn independently across agents from the distribution F . We write p q ( kA ) for the probability that type q chooses alternative kA , and p ( kA ) = ∫ Q p q ( kA ) d F for the overall share in the population. Similarly, we write p q ( dA ) for the probability that type q accepts the default, and p ( dA ) = ∫ Q p q ( dA ) d F for the population share. For each

q ∈ Q we have [ GLYPH&lt;229&gt; nA k = 1 p q ( kA )] + p q ( dA ) = 1, and likewise [ GLYPH&lt;229&gt; nA k = 1 p ( kA )] + p ( dA ) = 1 in aggregate. If wishing to emphasize the role of the type distribution in determining the choice probabilities, we write p ( kA ; F ) and p ( dA ; F ) .

The basic scenario of interest involves a large population choosing from a fixed menu M with | M | = nM = n ≥ 2. The analyst observes the aggregate choice shares, but knows neither the common preference order nor the distribution of cognitive types. In this context we shall generally suppress dependence on M , writing p q ( k ) and p q ( d ) for the type-specific frequencies and p ( k ) and p ( d ) for the population shares. Our goal is to deduce information about the distribution F from 〈 p ( 1 ) , p ( 2 ) , . . . , p ( n ) , p ( d ) 〉 , and to use this information to predict aggregate choices from menus other than M .

Weproceed now to specialize this framework to a more concrete model in which the cognitive heterogeneity relates to limited attention. Each agent will consider (i.e., pay attention to) a subset of the alternatives, and among those considered will choose the best option according to the common preference order. If the preference-maximizing alternative is not in the consideration set, this will result in a sub-optimal decision.

## 2.2 The consideration capacity model

Let g ∈ { 0, 1, 2, . . . } = Q denote a limit on the cardinality of the agent's consideration set; that is, the consideration capacity . When 1 ≤ g &lt; n we assume that the agent is equally likely to consider each G ⊂ M with | G | = g , and when g ≥ n we know with certainty that the entire menu M will be considered. In the former case there are ( n g ) candidate sets, of which ( n -k g -1 ) contain alternative k and do not contain any superior alternative glyph[lscript] &lt; k . For 1 ≤ g &lt; n , the probability of k being chosen is thus ( n -k g -1 ) / ( n g ) . Note that this probability is 0 for k &gt; n -g + 1, since here there are fewer than g -1 alternatives inferior to k that can populate the consideration set in order to allow k to be chosen. Of course, whenever the full menu is considered we know that alternative 1 will be chosen regardless of the value of g ≥ n . 8

8 We have assumed that the common preference relation glyph[followsorequal] is a linear order; i.e., that no two distinct alternatives are indifferent. If we allow for indifference then, defining w k ( R ) = |{ j : jRk }| , for 1 ≤ g &lt; n the probability of option k being chosen is [( w k ( glyph[followsorequal] ) g ) -( w k ( glyph[follows] ) g )][( n g )[ w k ( glyph[followsorequal] ) -w k ( glyph[follows] )]] -1 (with Equations 1, 6, and 14 below modified accordingly). While this generalization causes no significant difficulty for the

The type-conditional choice frequencies can now be expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Defining the probability masses

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the corresponding aggregate shares are then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that for 1 ≤ k &lt; n we can use Equation 6 to compute

<!-- formula-not-decoded -->

This relation shows that, when we move one ordinal step up the preference scale, the aggregate choice share increases for two reasons: Firstly, the k th best alternative can be chosen when g = n -k + 1, unlike the next best option. And secondly, for values of g smaller than this the better alternative is chosen more frequently, since there are more

derivation of choice shares, we shall nevertheless maintain the linear ordering assumption so as to avoid our main objective of cognitive identification being hampered by a feature of preferences alone. We also view the prohibition on indifference as relatively innocuous in the present, finite-menu setting.

ways of populating the rest of the consideration set with inferior options.

Note also that setting k = n in Equation 6 yields p ( n ) = p ( 1 ) n and hence

<!-- formula-not-decoded -->

Similarly, setting k = n -1 in Equation 8 yields p ( n -1 ) -p ( n ) = 2 p ( 2 ) n [ n -1 ] and hence

<!-- formula-not-decoded -->

Equations 9-10 prefigure the recursive method employed in Section 3 to identify the cognitive type distribution, in which the probabilities p ( 1 ) , . . . , p ( n -1 ) are deduced sequentially, with one additional choice share used at each step.

Finally, using Equation 9, we can write Equation 8 in terms of probability ratios as

<!-- formula-not-decoded -->

For instance, when k = n -1 we find that the probability mass ratio

<!-- formula-not-decoded -->

between the two smallest (nonzero) values of the consideration capacity depends only on the aggregate choice share ratio between the two worst alternatives on the menu.

## 2.3 Aspecial case: The consideration probability model

One special case of the consideration capacity model is a version of the consideration probability model studied by Manzini and Mariotti [30]. To see this, denote by r ∈ [ 0, 1 ] the probability that the agent considers each alternative on the menu, with consideration independent across agents and alternatives. Since the same consideration probability applies independently to each alternative, all subsets of the menu of a given size are equally likely to be the consideration set. Moreover, the probability of a consideration set of size g ≤ n is

<!-- formula-not-decoded -->

and clearly p ( g ) = 0 for g &gt; n . Adapting Equations 6-7 to this special case, we obtain the aggregate choice shares

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for the consideration probability model. As in the general case, alternative k 's choice share is the probability that this option and nothing better is considered, and the share of the default outcome is the probability that nothing at all is considered.

## 3 Inference from aggregate choices

## 3.1 Preference identification

In the context of our limited attention model, the agents' common preferences over the alternatives are fully revealed by the observed choice shares under weak conditions. Indeed, the following result is a simple consequence of Equation 8.

Proposition 1. For the consideration capacity model, with 1 ≤ k &lt; n:

- (i) p ( k ) ≥ p ( k + 1 ) .
- (ii) If GLYPH&lt;229&gt; n -k + 1 g = 2 p ( g ) &gt; 0 then p ( k ) &gt; p ( k + 1 ) .
- (iii) If p ( 2 ) &gt; 0 then p ( 1 ) &gt; p ( 2 ) &gt; · · · &gt; p ( n ) .

Using Equation 13, we can also specialize (iii) to the consideration probability model.

Corollary 1. For the consideration probability model, if the support of F intersects ( 0, 1 ) then p ( 1 ) &gt; p ( 2 ) &gt; · · · &gt; p ( n ) .

We conclude that, under the homogeneous tastes assumption, the preferences are for practical purposes fully revealed by aggregate choice data, and efforts can be focused squarely on the cognitive identification problem. For the remainder of Section 3 we assume tacitly that p ( 2 ) &gt; 0, ensuring that the choice shares p ( 1 ) &gt; p ( 2 ) &gt; · · · &gt; p ( n ) faithfully reflect the underlying preference order.

## 3.2 Cognitive identification: Parametric analysis

To examine the cognitive inference problem in its most concrete manifestation, we first consider several natural functional forms for the type distribution. Our aim here is to show that in such cases the cognitive parameters can be revealed in a straightforward fashion from a small number of appropriately selected choice-share observations. As well as increasing our familiarity with the limited-attention model under investigation, the examples below will serve to highlight non-obvious ways that aggregate choices can convey information about the cognitive type distribution.

For both the consideration capacity model and the special case of the consideration probability model, we consider simple one- and two-parameter distributions for the cognitive type.

Example 1. [ Poisson g ] For m &gt; 0, let the consideration capacity g have the Poisson distribution p ( g ) = m g g ! e -m for 0 ≤ g &lt; n . In this case Equation 7 yields default share p ( d ) = p ( 0 ) = e -m , and thus m = -log p ( d ) . Alternatively, Equation 12 yields

<!-- formula-not-decoded -->

and so m = [ n -1 ] [ p ( n -1 ) p ( n ) -1 ] . glyph[square]

Example 2. [ Pascal g ] For r ∈ { 1, 2, 3, . . . } and q ∈ ( 0, 1 ) , let the consideration capacity g have the Pascal (or 'negative binomial') distribution p ( g ) = ( g + r -1 g ) [ 1 -q ] r q g for 0 ≤ g &lt; n . Equation 12 then yields

<!-- formula-not-decoded -->

We have also

<!-- formula-not-decoded -->

and Equations 17-18 can be solved simultaneously for the parameters

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Example 3. [ uniform r ] For r min ∈ [ 0, 1 ) , let the consideration probability r be distributed uniformly on [ r min, 1 ] . Since F ( r ) = r -r min 1 -r min , Equation 14 becomes

<!-- formula-not-decoded -->

The first choice share is then p ( 1 ) = 1 + r min 2 , yielding the parameter r min = 2 p ( 1 ) -1. glyph[square]

Example 4. [ Beta r ] For a , b &gt; 0, let the consideration probability have the Beta distribution F ( r ) = 1 B ( a , b ) ∫ r 0 t a -1 [ 1 -t ] b -1 d t (where B is the Beta function). Here Equation 14 appears as

<!-- formula-not-decoded -->

The first two choice shares are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we can solve for the parameters

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that for parameterizations of the consideration capacity g we have used the choice shares p ( n ) and p ( n -1 ) , corresponding to the least attractive alternatives, to elicit information about the cognitive type distribution. In contrast, for parameterizations of the consideration probability r we have used p ( 1 ) and p ( 2 ) , corresponding to the most attractive options. This mirrors our elicitation procedure below in the nonparametric setting, where each mass p ( g ) is seen to depend on the choice shares of a group of sufficiently unattractive alternatives (cf., Equation 28), and each moment of the r -distribution is seen to depend on the shares of a sufficiently attractive group (cf., Equation 32).

## 3.3 Cognitive identification: Nonparametric analysis

## 3.3.1 The nonparametric inference problem

The examples in the previous section have shown a variety of ways that information about the cognitive type distribution can be encoded in the aggregate choice shares, depending on the specific functional form employed. With this introduction, we turn now to the general structure of the inference problem. We shall see that identification of the type distribution remains tractable for the consideration capacity model even without parametric assumptions. This is because the choice shares are linear functions of the probability masses p ( g ) , which are in turn linear functions of the moments mj of F when we specialize to the consideration probability model. Moreover, each linear system has a simple triangular structure that enables it to be solved recursively, using one additional choice share at each step.

In view of these features of the inference problem, we can decode the information about the cognitive capacity distribution encoded in the choice share data by inverting a triangular n × n matrix. This will yield the probability of each capacity value strictly less than n , and adding one more option to the menu will give us knowledge of one additional probability mass. In the consideration-probability setting, we can then invert a second triangular n × n matrix to deduce the first n raw moments of F from the capacity probabilities. Finally, well-established tools (specifically, sparse matrix theory

and maximum entropy methods) will permit us to approximate F from its moments with increasing precision as the size of the menu grows (see Section 3.4).

## 3.3.2 Recovering n probability masses

Absent parametric assumptions, the aggregate choice shares are given by Equation 6. These relations can be written together in matrix form as

<!-- formula-not-decoded -->

The upper anti-triangular and left-stochastic matrix C has a lower anti-triangular inverse, allowing us to write p = C -1 p . 9 Indeed, we can calculate the components of p explicitly as

<!-- formula-not-decoded -->

and of course p ( 0 ) = p ( d ) = 1 -GLYPH&lt;229&gt; n k = 1 p ( k ) . Observe that since p ( n ) = 1 -F ( n -1 ) , it is in fact the probabilities of the capacities g = 0, 1, . . . , n -1 that are revealed; and g = n cannot be disambiguated from higher values. Indeed, all capacities greater than or equal to the number of alternatives will always be behaviorally indistinguishable. We summarize our conclusions as follows.

Proposition 2. In the consideration capacity model, the probability masses p are uniquely determined by the aggregate choice shares p .

9 Amatrix is left (resp., right) stochastic if all entries are nonnegative and all columns (resp., all rows) sum to one.

## 3.3.3 Consideration probability: Recovering n moments

Returning to the special case of the consideration probability model, let us write the j th raw moment of the type distribution as mj = ∫ 1 0 r j d F . The binomial in Equation 13 can then be expanded to yield

<!-- formula-not-decoded -->

In matrix form, these relations appear as

<!-- formula-not-decoded -->

The upper triangular matrix Q has an upper triangular inverse, so we have

<!-- formula-not-decoded -->

Performing this calculation, the raw moments are given explicitly by

<!-- formula-not-decoded -->

We summarize our conclusions for the special case as follows.

Proposition 3. In the consideration probability model, the raw moments m are uniquely determined by the aggregate choice shares p .

## 3.4 Consideration probability: Beyond moments

## 3.4.1 From moments to distributions

Continuing to focus on the consideration probability model, we shall throughout Section 3.4 treat as known a finite number of raw moments of the cognitive type distribution F , appealing to Proposition 3 for justification. We proceed to outline two different strategies for ensuring that this moment information adequately captures F itself. The first strategy will rely on discreteness of the type distribution and ensure a unique characterization of F , while the second will rely on the existence of a density and guarantee convergence to F as n → ¥ .

## 3.4.2 Discrete type distributions

Suppose that F is a discrete distribution, with the consideration probability r taking on values 〈 r 1 , r 2, . . . , r L 〉 . The number L of cognitive types is known, though the values themselves may be unknown. We assume, however, that the values are located on a (known) finite grid of admissible points in [ 0, 1 ] , which can be as fine as desired.

The realized values of r have probabilities 〈 x ( r 1 ) , x ( r 2 ) , . . . , x ( r L ) 〉 , each strictly positive and together summing to one, so that the j th raw moment of F appears as

<!-- formula-not-decoded -->

Treating the first n moments as known, Equation 33 supplies a system of n equalities in 2 L unknowns; namely, the values r glyph[lscript] and their associated probabilities x ( r glyph[lscript] ) . This system can be solved for n sufficiently large, but it is not obvious that the solution will be unique.

Assume that the grid of admissible values for r is 〈 0, 1 N , 2 N , . . . , 1 〉 , with the fineness parameter N large relative to L . 10 Then F is a discrete distribution defined entirely by the probability masses 〈 x ( glyph[lscript] N ) 〉 N glyph[lscript] = 0 , of which exactly L glyph[lessmuch] N are nonzero. Recovering the distribution thus amounts to finding a solution x of the system

10 For notational simplicity we use an evenly spaced grid of admissible values, but this is not essential for our conclusions.

<!-- formula-not-decoded -->

with each component x ( glyph[lscript] N ) weakly positive and exactly L components strictly positive. Here V is a Vandermonde matrix with many more columns (i.e., grid points) than rows (known moments), implying an under-determined system. 11 But the number L of grid points actually used could in principle be larger or smaller than n .

A result of Cohen and Yeredor [15, Theorem 1] applies to precisely this situation, stating that Equation 34 has a unique solution if n ≥ 2 L . We conclude the following.

Proposition 4. In the consideration probability model, if F is a discrete distribution over L admissible types, with n ≥ 2 L, then F is uniquely determined by the aggregate choice shares p .

This result means that in practice any discrete distribution for the consideration probability r can be fully recovered from aggregate choice share data provided the number of alternatives is large relative to the number of cognitive types.

## 3.4.3 Type distributions with a density

Nowsuppose that the cognitive type distribution F admits a density f . In this case we will clearly not be able to recover F fully from a finite number n of moments. Instead, we aim to ensure that the known moments yield a reliable approximation of the true distribution.

Our method relies on standard techniques from the 'Hausdorff moment problem' for distributions on a closed interval. Adopting a maximum entropy approach, define

11 See, e.g., Macon and Spitzbart [28] for the definition and properties of Vandermonde matrices.

the n th approximating density ˆ fn as the solution to the optimization problem

<!-- formula-not-decoded -->

subject to the ( j th-moment) constraint

<!-- formula-not-decoded -->

for j = 0, 1, . . . , n . Mead and Papanicolaou [34, Theorem 2] show that such a solution exists and is unique; 12 and that for each bounded, continuous y : [ 0, 1 ] →glyph[Rfractur] we have

<!-- formula-not-decoded -->

Write ˆ Fn for the distribution function associated with the approximating density ˆ fn . For any menu A and each k ≤ min { n , | A |} , we now have that

<!-- formula-not-decoded -->

Here the first and third equalities follow from the observation that in the consideration probability model an alternative's choice share depends only on its rank on the menu according to the preference order. Moreover, in this model we have p = CQm and the shares of the n best alternatives are determined by the first n moments. The constraints in Equation 36 guarantee that these moments coincide for the distributions ˆ Fn and F , yielding the second equality in Equation 38. We summarize our findings as follows.

Proposition 5. In the consideration probability model, if F admits a density then there exists a map m ↦→ ˆ Fn such that:

- (i) The sequence 〈 ˆ Fn 〉 ¥ n = 1 converges weakly to F .
- (ii) For any menu A and each k ≤ min { n , | A |} , we have p ( kA ; ˆ Fn ) = p ( kA ; F ) .

12 Indeed, the solution takes the form ˆ fn ( r ) = exp [ -GLYPH&lt;229&gt; n j = 0 l j r j ] , where the quantities 〈 l j 〉 n j = 0 are the Lagrange multipliers on the constraints in Equation 36.

As already noted, the constraints in Equation 36 require each approximation ˆ Fn to be observationally indistinguishable from the true distribution F in the sense that they generate the same first n moments, and hence the same aggregate choice shares over menu M . Proposition 5 reinforces this by guaranteeing that the cognitive heterogeneity in the population is reflected in two additional ways: Firstly, as the size of the observed menu increases, our approximation approaches (in the sense of weak convergence) the true distribution of the consideration probability. And secondly, each approximation Fn matches the true F not just over M , but also over the n best alternatives on any other menu A about which we may wish to make predictions.

## 3.5 Unobserved default outcome

## 3.5.1 Conditional choice shares

In this section we consider the prospects for cognitive identification when the default outcome is unobserved. Under this assumption our data set consists of the aggregate shares p ( k ) = p ( k ) 1 -p ( d ) conditional on an active choice being made. Of course, any ratio of aggregate shares of the form ˜ p ( k , glyph[lscript] ) = p ( k ) p ( glyph[lscript] ) = p ( k ) p ( glyph[lscript] ) is unaffected by the conditioning, and so Equations 11-12 remain valid when restated in terms of the conditional shares and the associated probability masses p ( g ) = p ( g ) 1 -p ( 0 ) .

## 3.5.2 Parametric analysis

We begin by adapting each of the parametric examples introduced in Section 3.2 to the unobserved default scenario.

Example 1. [ Poisson g ; continued ] Here m = [ n -1 ] [ ˜ p ( n -1, n ) -1 ] , as above. glyph[square]

Example 2. [ Pascal g ; continued ] Equation 17 can be written as ˜ p ( n -1, n ) = q [ r + 1 ] n -1 + 1, and similarly from Equation 11 we obtain

<!-- formula-not-decoded -->

These equations can be solved simultaneously for the parameters

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Example 3. [ uniform r ; continued ] From Equation 21 we have both p ( 1 ) = 1 + r min 2 and p ( 2 ) = [ 2 r min + 1 ][ 1 -r min ] 6 . Hence ˜ p ( 1, 2 ) = 3 [ 1 + r min ] [ 2 r min + 1 ][ 1 -r min ] , and it follows that

<!-- formula-not-decoded -->

Example 4. [ Beta r ; continued ] Equations 23-24 yield ˜ p ( 2, 1 ) = b a + b + 1 , and likewise we can compute ˜ p ( 3, 2 ) = b + 1 a + b + 2 . Solving for the parameters, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 3.5.3 Recovering n -1 probability mass ratios

In the non-parametric setting, it is straightforward to adapt Equation 28 to the case of an unobserved default outcome. Indeed, for each g = 2, 3, . . . , n we have that

<!-- formula-not-decoded -->

Thus we can use the conditional choice shares to recover n -1 probability mass ratios, though without knowledge of the default share p ( d ) = p ( 0 ) we are of course unable to determine the masses themselves.

## 3.5.4 Consideration probability: Recovering n -1 moment ratios

For the special case of the consideration probability model, Equation 32 can likewise be adapted to the unobserved default scenario. Here, for each j = 2, 3, . . . , n , we have

<!-- formula-not-decoded -->

This yields n -1 ratios of raw moments, and we could proceed to use methods such as those in Section 3.4 above to approximate the shape of the cognitive type distribution F (the mean m 1 of which would remain undetermined without knowledge of the default share). 13

## 4 Preference heterogeneity

## 4.1 Known taste distribution

Section 3 has studied the identification properties of our model of consideration set formation under the assumption that preferences are homogeneous. We now aim to show that the preceding analysis can be extended to allow for heterogeneous preferences, provided the taste distribution is known and statistically independent of the cognitive (type) distribution. 14 We then proceed (in Section 4.2) to investigate the prospects for identification when both the taste and cognitive distributions are unknown.

To incorporate preference heterogeneity into the present framework, we order the alternatives arbitrarily as M = { 1, 2, . . . , n } and write j : M → { 1, 2, . . . , n } for the map that associates each option with its preference rank. 15 We enumerate the possible rankings as 〈 j h 〉 n ! h = 1 , write t h for the probability of j h , and denote by P ( h ) the n × n

13 Horan [24] considers an unobserved default outcome in the context of a dataset with choices from multiple menus, showing that the identification properties of Manzini and Mariotti's [30] independent random consideration model remain largely intact.

14 The distribution of taste parameters-such as discount factors or risk-aversion coefficients-may be treated as known for our purposes if these characteristics can be elicited from agents separately, in a setting (e.g., a laboratory experiment) where limited attention is thought to be irrelevant or controllable to an acceptable degree.

15 This formulation maintains the assumption of linear order preferences imposed in Section 2.

permutation matrix corresponding to j h . 16 With preference heterogeneity, Equation 27 then becomes

<!-- formula-not-decoded -->

Provided the 'average preference permutation matrix' B = GLYPH&lt;229&gt; n ! h = 1 t h P ( h ) has full rank, it follows that p = [ BC ] -1 p , and similarly Equation 31 becomes m = [ BCQ ] -1 p for the special case of the consideration probability model. We conclude that the aggregate choice shares can still be used to find the probability masses in p and the raw moments in m , as long as the taste distribution yields a nonsingular matrix B .

As a convex combination of permutation matrices, the average preference permutation B is always bistochastic. 17 Clearly there exist taste distributions t = 〈 t h 〉 n ! h = 1 for which this matrix is not invertible; e.g., the uniform distribution (with each t h = 1 n ! ) yields a singular B (with each entry equal to 1 n ). However, invertibility is the generic situation, implying the following extensions of Propositions 2-3. 18

Proposition 6. In the consideration capacity model with known preference heterogeneity, for almost all taste distributions t the probability masses p are uniquely determined by the aggregate choice shares p .

Proposition 7. In the consideration probability model with known preference heterogeneity, for almost all taste distributions t the raw moments m are uniquely determined by the aggregate choice shares p .

The following example illustrates the handling of known preference heterogeneity in the context of the consideration capacity model.

Example 5. [ exploded logit ] Let n = 3, define u : M → glyph[Rfractur] by u ( k ) = log k , and suppose that the distribution of tastes is determined by an exploded logit based on u . For

16 More explicitly, the permutation matrix P ( h ) translates the k th row of an n × n target matrix A into the j h ( k ) th row of the product P ( h ) A . Similarly, postmultiplying by P ( h ) permutes the columns of A .

17 A matrix is bistochastic if it is both left and right stochastic. The Birkhoff-von-Neumann Theorem states that the class of n × n bistochastic matrices is the convex hull of the set of n × n permutation matrices.

18 Observe that det ( B ) is a polynomial function of t ∈ glyph[Rfractur] n ! , and recall that any real-valued polynomial function on a Euclidean space is either identically zero or nonzero almost everywhere (see, e.g., Caron and Traynor [12]). Since det ( B ) is nonzero for the case of homogeneous preferences, it is not identically zero, and thus B is generically invertible.

instance, the probability assigned to the ranking j 2 given by 2 glyph[follows] 3 glyph[follows] 1 is calculated as

<!-- formula-not-decoded -->

The average preference permutation matrix is then

<!-- formula-not-decoded -->

which is nonsingular (with det ( B ) = -1 30 ). Now it is straightforward to compute

<!-- formula-not-decoded -->

and as always p ( 0 ) = p ( d ) . glyph[square]

## 4.2 Unknown taste distribution

Continuing to allow for heterogeneous preferences, we next consider the problem of identifying the cognitive distribution when the taste distribution too is unknown. Here the information in a single observation of aggregate choices is clearly insufficient to reveal both distributions nonparametrically. Indeed, Propositions 2-3 already consume all n degrees of freedom in order to infer probability masses or raw moments of F . The impracticality of deducing cognition and tastes simultaneously from our basic dataset is illustrated in the following simple example.

Example 6. Let n = 2 and j 1 ( 1 ) = 1, so that t 1 is the probability of the ranking 1 glyph[follows] 2. Equation 47 then takes the form

<!-- formula-not-decoded -->

an underdetermined system in which the cognitive distribution 〈 p ( 1 ) , p ( 2 ) 〉 and the taste distribution t 1 cannot be disambiguated. glyph[square]

To gain some leverage on the unknown tastes scenario it will be necessary to relax the stringent assumption that our dataset consists of aggregate choice shares from a single menu, and a variety of relaxations are possible. 19 The approach we shall adopt here is to suppose that the researcher has access to choice data from the same population of agents on multiple 'occasions' across which the cognitive distribution is stable. While we assume for convenience that the size of the menu is constant, the alternatives themselves need not be identical across occasions. For instance, the objects of choice could be interpreted as the same physical items at time-varying prices; the current model of a product offered in successive periods by a fixed set of suppliers; or the options available in an experiment with multiple rounds or treatments.

Weassumefurther that our dataset consists of the joint distribution of choices across occasions; as arising, for example, from discrete choice panel data or from a sequence of discrete choice experiments. Although joint choice shares of this sort comprise 'aggregate' data only from a somewhat literalist point of view, the agents in the population can remain anonymous in the sense that no observations on individuals will be required for our analysis other than their observed choices. 20

The advantage of this new multi-occasion setting is that it will allow us to deploy a powerful mathematical result on tensor decompositions to determine the cognitive

19 One strategy would be to supply the researcher with aggregate data on choices from multiple subsets of the menu (cf. Geng and Ozbay [21]), while assuming stable tastes. Another strategy-explored in an earlier version of this paper-would be to supplement the dataset with covariates and estimate a random utility model of preference determination.

20 With I occasions and n alternatives, a single agent's joint choice can be described by a unit vector in n I -dimensional space. The aggregate choice frequencies for the population are then given by the sum of these vectors, which is equivalent to the aggregate joint distribution of choices in our dataset.

distribution even in the context of unknown and possibly changing tastes. Indeed, we shall find (in Proposition 8) that joint choice share data from as few as three occasions suffices generically to infer the consideration capacity distribution in full as well as substantial information about the distribution of tastes.

Formally, we index the occasions by i = 1, . . . , I and suppose that on each occasion our population of agents chooses from a menu M = { 1, . . . , n } with default d / ∈ M . Here neither k ∈ M nor the default d need represent the same economic outcome on different occasions, but the cardinality n of the menu is assumed to be constant. 21 The taste distribution on occasion i is denoted by t i = 〈 t ih 〉 n ! h = 1 , and agents are assumed to retain their cognitive types across occasions so that the distribution F is stable. We write p q ( k 1 · · · kI ) for the joint probability that on each occasion i an individual of type q chooses alternative ki . Our dataset then consists of the corresponding population shares p ( k 1 · · · kI ) = ∫ Q p q ( k 1 · · · kI ) d F , and as before our objective is to use this data to deduce information about the underlying cognitive distribution F .

In the context of the consideration capacity model, we assume that the realizations of the consideration set G and the preference ranking j h are independent across occasions conditional on the (capacity) type g . The analog of Equation 1 is then

<!-- formula-not-decoded -->

where (for 1 ≤ g &lt; n ) the product is over the various occasions i , the outer sum is over the possible ranking positions r of the chosen alternative ki , and the inner sum is over the rankings that place ki in position r . Now the analog of Equation 6 appears as

<!-- formula-not-decoded -->

showing explicitly how the population choice shares are determined by the cognitive

21 This assumption simplifies our notation considerably, but is not essential for the analysis below.

distribution p in conjunction with the taste distributions 〈 t i 〉 I i = 1 .

The following example illustrates the multi-occasion framework and demonstrates how joint choice share data can be used to infer the cognitive and taste distributions.

Example 7. Let n = 2, I = 3, and each j i 1 ( 1 ) = 1. The joint probabilities are the weighted sums of the two type-conditional joint probabilities, that is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and so on. For t 11 , t 21 , t 31 ∈ ( 0, 1 ) , we can combine the four equations above to yield

<!-- formula-not-decoded -->

enabling recovery of the probability mass

<!-- formula-not-decoded -->

Equation 58 and two analogous relations then yield the three taste distributions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, from Equation 54 (or any of the other joint probability expressions) we obtain

<!-- formula-not-decoded -->

We aim now to demonstrate that the cognitive identification seen in Example 7 is a generic feature of the multi-occasion setting. To this end, we condition on the event g &gt; 0 and represent our dataset as a tensor S of order I with dimensions n ×··· × n and entries Sk 1 ,..., k I = p ( k 1 ··· k I ) 1 -p ( d ··· d ) . 22 Writing B i = GLYPH&lt;229&gt; n ! h = 1 t ih P ( h ) for the average preference permutation matrix on occasion i , we can then express Equation 53 more compactly as

<!-- formula-not-decoded -->

where ⊗ is the outer product operator and 1 g the unit vector for component g . 23

Equation 64 decomposes the joint shares into a linear combination of n rank-1 tensors. The uniqueness properties of such decompositions have been studied extensively, with Kruskal [25, Theorem 4a] supplying a fundamental theorem that has been elucidated and refined by Sidiropoulos and Bro [40], Allman et al. [3], and Rhodes [36], among others. We shall use a special case of the result due to Rhodes [36, Corollary 2], adapted for our setting as follows.

Lemma 1. [Kruskal; Rhodes] Given any collection 〈 Z 1 , Z 2, Z 3 〉 of invertible n × n matrices, the tensor T = GLYPH&lt;229&gt; n g = 1 [ Z 1 1 g ⊗ Z 2 1 g ⊗ Z 3 1 g ] uniquely determines each Z i up to column rescaling and permutation. That is, for any 〈 ˆ Z 1 , ˆ Z 2, ˆ Z 3 〉 such that GLYPH&lt;229&gt; n g = 1 [ ˆ Z 1 1 g ⊗ ˆ Z 2 1 g ⊗ ˆ Z 3 1 g ] = T there exist invertible diagonal matrices 〈 D 1 , D 2, D 3 〉 and a permutation matrix P such that D 1 D 2 D 3 = I n and each ˆ Z i = Z i D i P . 24

22 Atensor is a multidimensional array that generalizes the concept of a matrix to allow for an arbitrary number of indices-this number being the order of the tensor. The dimensions of a tensor indicate the number of possible values of each index, generalizing the number of rows and columns of a matrix.

23 Recall that the outer product of a pair of vectors is the first multiplied by the transpose of the second, and similarly each further outer product operation adds another dimension to the resulting array. A tensor is said to be of rank 1 if it is an outer product of vectors.

24 Here I n denotes the n × n identity matrix. Note that the actual result in Rhodes [36] is substantially more general than this statement, since he allows the Z i matrices to have different numbers of rows and

Setting I = 3 and applying this tool to our joint choice share tensor S , we can show generic cognitive identification in the multi-occasion environment.

Proposition 8. In the consideration capacity model with unknown preference heterogeneity and three occasions, if p glyph[greatermuch] 0 then for almost all taste distributions 〈 t 1 , t 2, t 3 〉 the probability masses p and average preference permutation matrices 〈 B 1 , B 2, B 3 〉 are uniquely determined by the joint choice shares p ( k 1 k 2 k 3 ) for 1 ≤ k 1 , k 2, k 3 ≤ n. 25

Proof. Writing D ( p ) for the diagonal matrix with entries p = 〈 p ( g ) 〉 n g = 1 glyph[greatermuch] 0, we can (following [3, p. 3118]) set Z 1 = [ B 1 C ] D ( p ) , Z 2 = B 2 C , and Z 3 = B 3 C , whereupon

<!-- formula-not-decoded -->

Here each B i C has full rank for almost all taste distributions t i (see Footnote 18), and since p glyph[greatermuch] 0 it follows that each Z i is invertible. For duplicate parameters 〈 ˆ B 1 , ˆ B 2, ˆ B 3 〉 and ˆ p glyph[greatermuch] 0 such that the corresponding GLYPH&lt;229&gt; n g = 1 [ ˆ Z 1 1 g ⊗ ˆ Z 2 1 g ⊗ ˆ Z 3 1 g ] = S , Lemma 1 ensures that there exist rescalings 〈 D 1 , D 2, D 3 〉 and a permutation P such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Writing the vector of ones as 1 = GLYPH&lt;229&gt; n g = 1 1 g , we have 1 glyph[latticetop] ˆ B i C = 1 glyph[latticetop] [ B i C ] D i P for i = 2, 3 from Equations 67-68. Since B i and ˆ B i are bistochastic and C is left stochastic, it follows that 1 glyph[latticetop] = 1 glyph[latticetop] D i P and thus 1 glyph[latticetop] = 1 glyph[latticetop] P glyph[latticetop] = 1 glyph[latticetop] D i . We conclude that D 2 = D 3 = I n , and hence D 1 = [ D 2 D 3 ] -1 = I n as well.

Similarly, we have ˆ p glyph[latticetop] = 1 glyph[latticetop] D ( ˆ p ) = 1 glyph[latticetop] D ( p ) P = p glyph[latticetop] P from Equation 66 and hence

one of them to have linearly dependent columns. This necessitates an additional hypothesis imposing a lower bound on the 'Kruskal rank' (see [36, p. 1819]) of the Z i matrix in question-a hypothesis that is trivially satisfied in the square, full-rank case.

25 Our result shows that three occasions suffice (generically) for determination of p , and data from additional occasions will neither help nor hinder cognitive identification. Extensions of Kruskal's theorem to tensors of order higher than three have been studied; e.g., by Sidiropoulos and Bro [40, Theorem 3]. We do not pursue this extension at present, since it is orthogonal to our main goal of inferring cognitive heterogeneity from minimal data.

ˆ p = P glyph[latticetop] p . It follows that D ( ˆ p ) = D ( P glyph[latticetop] p ) = P glyph[latticetop] D ( p ) P , so that Equation 66 yields [ ˆ B 1 C ] P glyph[latticetop] D ( p ) P = [ B 1 C ] D ( p ) P and [ ˆ B 1 C ] P glyph[latticetop] = B 1 C . Together with Equations 67-68, this shows that each ˆ B i = B i [ CPC -1 ] . The duplicate parameters 〈 ˆ B 1 , ˆ B 2, ˆ B 3 〉 and ˆ p are thus seen to result from label swapping; i.e., a garbling of the type distribution p (via the permutation P glyph[latticetop] ) that is reversed by adjustments to the taste distributions B i . When labels are assigned correctly we have P = I n , so that ˆ p = p and each ˆ B i = B i .

Observe that the joint choice share data do not fully determine the taste distributions 〈 t 1 , t 2, t 3 〉 . Indeed, factorial explosion of the number of rankings of n alternatives makes it obvious that such identification cannot be possible in general. In relation to tastes, what the joint choice shares determine are the average preference permutation matrices 〈 B 1 , B 2, B 3 〉 , which record the probabilities GLYPH&lt;229&gt; h : j h ( k i )= r t ih of alternative ki being ranked in position r on occasion i . Thus the remaining scope for non-uniqueness is limited to shifts between rankings that preserve these marginal position probabilities.

The identification of the cognitive distribution and the average preference permutations in Proposition 8 requires no parametric assumptions on the primitives of the consideration capacity model. Introducing such assumptions may enable us to refine the taste distribution beyond the probabilities in each B i ; a task that is simplified by the problem now amounting to one of determining the preferences of agents with known cognitive type. In fact, since the type-conditional choice distributions have already been recovered, we could simply focus on the behavior of full-attention types (with g ≥ n ) and apply known techniques to elicit the distribution of preferences on each occasion. For instance, we could assume that the type-conditional choices result from a random utility model (RUM) with a known error distribution, or by a single-crossing RUM as defined in Apesteguia et al. [5]. In any event, such parametric assumptions would be unrelated to the limited-attention aspects of the model and unnecessary for achieving our primary goal in this section, which is to identify the cognitive distribution in the presence of unknown taste heterogeneity.

## 5 Conclusion

In this paper we have shown how aggregate choice shares can identify the distribution of cognitive characteristics in a population of agents exhibiting limited attention. A key advantage of our theory is that it uses minimal data: either choice shares from a single menu (for a known taste distribution), or joint shares from as few as three menus (for an unknown taste distribution). This contrasts with prior theoretical work on bounded rationality, much of which uses individual choices from a family of overlapping menus. Both approaches can be brought to bear on data, but in our view the present framework is better suited to the practice of empirical research on discrete choice.

Under homogeneous tastes, we find that both the consideration capacity model and the special case of the consideration probability model are highly tractable in terms of cognitive identification. In each model the aggregate choice shares are seen to be linear in quantities that are highly informative about the cognitive distribution; respectively, small capacity probabilities and low-order raw moments. The resulting linear systems are recursive and easily solved for the quantities in question. Indeed, our main findings demonstrate that for large menus the cognitive distribution is essentially fully identified, and even for smaller menus we can infer important features of this distribution (Propositions 2-5). These findings extend generically to heterogeneous tastes with a known distribution, and when the taste distribution is unknown a parsimonious extension to our dataset ensures generic cognitive identification (Propositions 6-8).

Finally, we mention three possible ways to build on the work reported in this paper. One is to generalize the models of consideration set formation that we have studied; for instance, by relaxing the assumption that each G with | G | = g is equally likely to occur in the consideration capacity model, or by allowing r to vary by alternative in the consideration probability model (see Brady and Rehbeck [9]). A second is to bring additional models of bounded rationality-incorporating phenomena such as framing effects or satisficing-into the present setting. And a third is to develop a complete econometric specification of the multi-occasion environment in Section 4.2 that can be used to estimate preference and attention characteristics from joint choice share data.

## References

- [1] Jason Abaluck and Abi Adams (2017). What do consumers consider before they choose? Identification from asymmetric demand responses. Unpublished.
- [2] Victor H. Aguiar, Maria Jose Boccardi, Nail Kashaev, and Jeongbin Kim (2018). Does random consideration explain behavior when choice is hard? Evidence from a large-scale experiment. Unpublished.
- [3] Elizabeth S. Allman, Catherine Matias, and John A. Rhodes (2009). Identifiability of parameters in latent structure models with many observed variables. Annals of Statistics 37:3099-3132.
- [4] Jose Apesteguia and Miguel A. Ballester (2013). Choice by sequential procedures. Games and Economic Behavior 77:90-99.
- [5] Jose Apesteguia, Miguel A. Ballester, and Jay Lu (2017). Single-crossing random utility models. Econometrica 85:661-674.
- [6] Nicholas Baigent and Wulf Gaertner (1996). Never choose the uniquely largest: A characterization. Economic Theory 8:239-249.
- [7] Levon Barseghyan, Maura Coughlin, Francesca Molinari, and Joshua Teitelbaum (2018). Heterogeneous consideration sets and preferences. Unpublished.
- [8] Levon Barseghyan, Francesca Molinari, and Matthew Thirkettle (2019). Discrete choice under risk with limited consideration. Unpublished.
- [9] Richard L. Brady and John Rehbeck (2016). Menu-dependent stochastic feasibility. Econometrica 84:1203-1223.
- [10] Andrew Caplin and Mark Dean (2015). Revealed preference, rational inattention, and costly information acquisition. American Economic Review 105:2183-2203.
- [11] Andrew Caplin, Mark Dean, and Daniel Martin (2011). Search and satisficing. American Economic Review 101:2899-2922.

- [12] Richard Caron and Tim Traynor (2005). The zero set of a polynomial. Unpublished.
- [13] Matias D. Cattaneo, Xinwei Ma, Yusufcan Masatlioglu, and Elchin Suleymanov (2017). A random attention model. Unpublished.
- [14] Vadim Cherepanov, Timothy Feddersen, and Alvaro Sandroni (2013). Rationalization. Theoretical Economics 8:775-800.
- [15] Anna Cohen and Arie Yeredor (2011). On the use of sparsity for recovering discrete probability distributions from their moments. Proceedings of the 2011 IEEE Statistical Signal Processing Workshop .
- [16] Gregory S. Crawford, Rachel Griffith, and Alessandro Iaria (2018). Preference estimation with unobserved choice set heterogeneity using sufficient sets. Unpublished.
- [17] Geoffroy de Clippel, Kfir Eliaz, and Kareen Rozen (2014). Competing for consumer inattention. Journal of Political Economy 122:1203-1234.
- [18] Henrique de Oliveira, Tommaso Denti, Maximilian Mihm, and Kemal Ozbek (2017). Rationally inattentive preferences and hidden information costs. Theoretical Economics 12:621-654.
- [19] Federico Echenique, Kota Saito, and Gerelt Tserenjigmid (2017). The perceptionadjusted Luce model. Mathematical Social Sciences 93:67-76.
- [20] Martin Gaynor, Carol Propper, and Stephan Seiler (2016). Free to choose? Reform, choice, and consideration sets in the English National Health Service. American Economic Review 106:3521-3557.
- [21] Sen Geng and Erkut Y. Ozbay (2018). Choice with limited capacity. Unpublished.
- [22] Justine S. Hastings, Ali Hortac ¸su, and Chad Syverson (2017). Sales force and competition in financial product markets: The case of Mexico's social security privatization. Econometrica 85:1723-1761.

- [23] Elisabeth Honka, Ali Hortac ¸su, and Maria Ana Vitorino (2017). Advertising, consumer awareness, and choice: Evidence from the U.S. banking industry. Rand Journal of Economics 48:611-646.
- [24] Sean Horan (2018). Random consideration and choice: A case study of 'default' options. Unpublished.
- [25] Joseph B. Kruskal (1977). Three-way arrays: Rank and uniqueness of trilinear decompositions, with application to arithmetic complexity and statistics. Linear Algebra and its Applications 18:95-138.
- [26] Zhentong Lu (2016). Estimating multinomial choice models with unobserved choice sets. Unpublished.
- [27] R. Duncan Luce (1959). Individual Choice Behavior: A Theoretical Analysis . Wiley.
- [28] Nathaniel Macon and Abraham Spitzbart (1958). Inverses of Vandermonde matrices. The American Mathematical Monthly 65:95-100.
- [29] Paola Manzini and Marco Mariotti (2007). Sequentially rationalizable choice. American Economic Review 97:1824-1839.
- [30] Paola Manzini and Marco Mariotti (2014). Stochastic choice and consideration sets. Econometrica 82:1153-1176.
- [31] Yusufcan Masatlioglu and Daisuke Nakajima (2013). Choice by iterative search. Theoretical Economics 8:701-728.
- [32] Yusufcan Masatlioglu, Daisuke Nakajima, and Erkut Y. Ozbay (2012). Revealed attention. American Economic Review 102:2183-2205.
- [33] Daniel McFadden (2001). Economic choices. American Economic Review 91:351-378.
- [34] Laurence R. Mead and Nikos Papanicolaou (1984). Maximum entropy in the problem of moments. Journal of Mathematical Physics 25:2404-2417.

- [35] Efe A. Ok, Pietro Ortoleva, and Gil Riella (2014). Revealed (p)reference theory. American Economic Review 105:299-321.
- [36] John A. Rhodes (2010). A concise proof of Kruskal's theorem on tensor decomposition. Linear Algebra and its Applications 432:1818-1824.
- [37] John H. Roberts and James M. Lattin (1997). Consideration: Review of research and prospects for future insights. Journal of Marketing Research 34:406-410.
- [38] Yuval Salant and Ariel Rubinstein (2008). ( A , f ) : Choice with frames. Review of Economic Studies 75:1287-1296.
- [39] Allan Shocker, Moshe Ben-Akiva, Bruno Boccara, and Prakash Nedungadi (1991). Consideration set influences on consumer decision making and choice: Issues, models, and suggestions. Marketing Letters 2:181-198.
- [40] Nicholas D. Sidiropoulos and Rasmus Bro (2000). On the uniqueness of multilinear decomposition of N-way arrays. Journal of Chemometrics 14:229-239.
- [41] Christopher A. Sims (2003). Implications of rational inattention. Journal of Monetary Economics 50:665-690.
- [42] Michelle Sovinsky Goeree (2008). Limited information and advertising in the U.S. personal computer industry. Econometrica 76:1017-1074.
- [43] Kenneth E. Train (2009). Discrete Choice Methods with Simulation . Cambridge University Press.
- [44] Christopher J. Tyson (2008). Cognitive constraints, contraction consistency, and the satisficing criterion. Journal of Economic Theory 138:51-70.
- [45] Christopher J. Tyson (2013). Behavioral implications of shortlisting procedures. Social Choice and Welfare 41:941-963.
- [46] Erjen van Nierop, Bart Bronnenberg, Richard Paap, Michel Wedel, and Philip Hans Franses (2010). Retrieving unobserved consideration sets from household panel data. Journal of Marketing Research 47:63-74.