## Stochastic Revealed Preferences with Measurement Error ∗

Victor H. Aguiar vaguiar@uwo.ca

†

Nail Kashaev † nkashaev@uwo.ca

First version: September 2017

This version: June 2020

Abstract A long-standing question about consumer behavior is whether individuals' observed purchase decisions satisfy the revealed preference (RP) axioms of the utility maximization theory (UMT). Researchers using survey or experimental panel data sets on prices and consumption to answer this question face the well-known problem of measurement error. We show that ignoring measurement error in the RP approach may lead to overrejection of the UMT. To solve this problem, we propose a new statistical RP framework for consumption panel data sets that allows for testing the UMT in the presence of measurement error. Our test is applicable to all consumer models that can be characterized by their first-order conditions. Our approach is nonparametric, allows for unrestricted heterogeneity in preferences, and requires only a centering condition on measurement error. We develop two applications that provide new evidence about the UMT. First, we find support in a survey data set for the dynamic and time-consistent UMT in single-individual households, in the presence of nonclassical measurement error in consumption. In the second application, we cannot reject the static UMT in a widely used experimental data set in which measurement error in prices is assumed to be the result of price misperception due to the experimental design. The first finding stands in contrast to the conclusions drawn from the deterministic RP test of Browning (1989). The second finding reverses the conclusions drawn from the deterministic RP test of Afriat (1967) and Varian (1982).

JEL classification numbers: C60, D10.

Keywords: rationality, utility maximization, time consistency, revealed preferences, measurement error.

∗ We thank the editor and three anonymous referees for comments and suggestions that have greatly improved the manuscript. We thank for their comments seminar and conference participants at UC Davis (ARE Department), University of Toronto, Université libre de Bruxelles, Caltech (DHSS), Arizona State University, Cornell University, NASMES 2018, MEG Meeting 2018, IAAE Conference 2018, BRIC 2018, CEA Meeting 2019, SAET Conference 2019, and the Conference on Counterfactuals with Economic Restrictions at UWO. We are grateful to Roy Allen, Andres Aradillas-Lopez, Elizabeth Caucutt, Laurens Cherchye, Ian Crawford, Tim Conley, Mark Dean, Rahul Deb, Thomas Demuynck, Federico Echenique, Yuichi Kitamura, Lance Lochner, Jim MacGee, Nirav Mehta, Salvador Navarro, Joris Pinkse, David Rivers, Bram De Rock, Susanne Schennach, Jörg Stoye, Al Slivinsky, Dan Silverman, and Todd Stinebrickner for useful comments and encouragement. We thank Vered Kurtz-David for providing validation data; Lance Lochner, Compute Ontario, and Compute Canada for computational resources; and the Social Sciences and Humanities Research Council for financial support.

† Department of Economics, University of Western Ontario.

## 1. Introduction

One well-known feature of consumer panel data sets-whether they are based on surveys, experiments, or scanners-is measurement error in prices or consumption. 1 This is of significant concern because measurement error is responsible for one important limitation of the standard revealed-preference (RP) framework. Specifically, RP tests tend to overreject the utility maximization theory (UMT). To the best of our knowledge, this paper proposes the first fully nonparametric statistical RP framework for consumer panel data sets in the presence of measurement error. We show that taking measurement error into account can significantly change the conclusions about the validity of the UMT in a given context. In the two applications we develop, we cannot reject the validity of the UMT, a finding which contradicts the conclusions of the deterministic RP framework.

Measurement error is the difference between the unobserved but true value of the variable of interest and its observed but mismeasured counterpart. If the UMT is valid, the corresponding RP conditions must be satisfied by the true prices and consumption. However, there is no reason to believe that either the mismeasured consumption or mismeasured prices that we usually observe are consistent with the RP conditions. A key concern for RP practitioners is that, in the presence of measurement error, a deterministic RP test may overreject the null hypothesis that the UMT is valid. We provide Monte Carlo evidence that this concern may be relevant in practice. Measurement error in consumption may arise in survey data due to misreporting, in experimental data due to trembling-hand errors, and in scanner data due to recording errors. 2 Measurement error in prices may arise in experimental data due to misperception errors and in scanner data due to unobserved coupons. 3

Our methodology covers as special cases the static UMT, and the classic dynamic UMT with exponential discounting. 4 When we apply our methodology to a consumer panel survey of Spanish households and allow for measurement error in consumption, we find that the dynamic UMT with exponential discounting cannot be rejected for single-individual households. This first finding contradicts the conclusions of the deterministic RP test of Browning (1989). When we apply our framework to a widely used experimental data set (Ahn et al., 2014) and allow for measurement error in prices that arises due to misperception, we find that the static UMT cannot be rejected. This second finding is the opposite of the conclusions that must be drawn when one applies the deterministic RP test of Afriat (1967) and Varian (1982) to the same data set. Taken together, these findings suggest that the negative conclusions about the validity of the UMT drawn from

1 See Mathiowetz et al. (2002), Echenique et al. (2011), Carroll et al. (2014), and Gillen et al. (2017).

2 Trembling-hand errors are nonsystematic mistakes incurred by a subject when trying to implement a decision because she has difficulties with the interface of an experiment.

3 Misperception errors are nonsystematic mistakes incurred by a subject because she misperceives information due to the experimental design.

4 We also cover firm-cost minimization (Varian, 1984), dynamic rationalizability with quasi-hyperbolic discounting (Blow et al., 2017), homothetic rationalizability (Varian, 1985), quasilinear rationalizability (Brown &amp; Calsamiglia, 2007), expected utility maximization (Diewert, 2012), and static utility maximization with nonlinear budget constraints (Forges &amp; Minelli, 2009).

the deterministic RP framework may not be robust to measurement error.

The leading solution for dealing with measurement error in the RP framework usually consists of perturbing (minimally) any observed individual consumption streams in order to satisfy the conditions of an RP test (Adams et al., 2014). However, this approach does not allow for standard statistical hypothesis testing. In particular, one cannot control the probability of erroneously rejecting a particular model when such a rejection could be an artifact of noisy measurements. Other works on RP with measurement error, such as the seminal contribution of Varian (1985), allow for statistical hypothesis testing but require knowledge of the distribution of measurement error; this may be impractical because this does not align with the nonparametric nature of the RP framework. In contrast, our procedure does not suffer from these issues.

Our main result is the formulation of a statistical test for the null hypothesis that a random data set of mismeasured prices and consumption is consistent with any given model that can be characterized by first-order conditions. Based on this test we provide and implement a general methodology to make out-of-sample predictions or counterfactual analyses with minimal assumptions (e.g., sharp bounds for average or quantile demand). Our approach takes advantage of the work of Schennach (2014) on Entropic Latent Variable Integration via Simulation (ELVIS) to provide a practical implementation of our test.

Our RP methodology is fully nonparametric and admits unrestricted heterogeneity in preferences. In addition, we require only a centering condition on the unobserved measurement error. The centering condition captures the application-specific knowledge we have about measurement error. Moreover, our framework is general enough to allow for (i) nonclassical measurement errors in consumption in survey environments, (ii) trembling-hand errors in experimental setups, (iii) misperception of prices due to experimental designs (price measurement error), and (iv) different forms of measurement error in prices in scanner environments.

In our first application in particular, we require that consumers be accurate, on average, in recalling and reporting their total expenditures. This assumption is compatible with systematic misreporting of consumption in surveys. For our second application, to an experimental data set, we require that errors in consumption or prices be centered around zero, which is compatible with trembling-hand and misperception errors in subjects' behavior. Measurement error in experimental data sets may arise because the experimental design may fail to elicit the intended choices of consumers. In such a case, classical measurement-error assumptions that allow for nonsystematic mistakes must be taken into account to ensure the external validity of the conclusions drawn from applying any RP test to this type of data set.

Our empirical contribution is to apply our methodology to a well-known consumer panel survey of single-individual and couples' households in Spain 5 in order to test for the dynamic UMT, and to reexamine the static UMT in a widely used experimental data set (Ahn et al., 2014); in doing this reexamination, we allow (separately) for trembling-hand errors and misperceived prices.

For the first application, we note that under the exponential discounting model, the consumer's time preferences are captured by a time-invariant discount factor and a time-invariant instanta-

5 This data set has been used in Beatty &amp; Crawford (2011), Adams et al. (2014), and Blow et al. (2017).

neous utility. The main feature of this model is the time-consistency of the exponential discounting consumer. In other words, if the consumer prefers consumption bundle c at time t to x at time t + k , then she will always prefer c at time τ to x at time τ + k . The exponential discounting model remains the workhorse of a large body of applied work in economics. However, several authors, such as Browning (1989), DellaVigna &amp; Malmendier (2006), and Blow et al. (2017), have provided suggestive evidence against the validity of this model. Our methodology addresses, in a nonparametric fashion, the presence of measurement error in survey data, in order to examine the robustness of these findings.

We find support for exponential discounting behavior for single-individual households. This contrasts with the results of applying the deterministic methodology of Browning (1989) to the same sample. At the same time, in line with the findings of Blow et al. (2017) (who also use the deterministic methodology of Browning, 1989), we reject the null hypothesis of exponential discounting for the case of couples. When compared with the single-household evidence, these results suggest that time inconsistencies in consumer behavior in the couples' case arise due to preference aggregation.

In our second application, we apply our methodology to test for the validity of the classical static UMT. Since the work of Afriat (1967) and Varian (1982), researchers have used the deterministic RP framework to examine the validity of the UMT in experimental data sets. The experimental design proposed by Ahn et al. (2014) is particularly useful for this task since it provides a controlled environment with substantial price variation, variation that guarantees that the UMT has empirical bite.

When we apply the deterministic RP test to this experimental data set, we conclude that the UMT is rejected for most subjects. Nonetheless, the external validity of the conclusions drawn from the deterministic RP tests applied to these experimental data sets may be limited. One key reason for this is that the elicitation of consumer behavior may have been subject to measurement error. Gillen et al. (2017) argue that experimental elicitations of choices are subject to random variation in participants' perception and focus. Moreover, RP practitioners since Afriat (1967) have recognized that the deterministic RP test for the static UMT may be too demanding in the presence of imperfect devices for the elicitation of choices. Many researchers have studied how to allow for optimization mistakes in the RP framework and how to measure the intensity of any departure from rationality. 6 However, none of the existing approaches designed to introduce the possibility of mistakes in the RP framework has allowed for a fully nonparametric approach to doing standard statistical hypothesis testing. In our application, we allow for the possibility of nonsystematic mistakes by requiring that the measurement error in consumption or prices be mean zero.

We cannot reject the null hypothesis of the validity of the static UMT with misperception of prices. However, when we allow only for trembling-hand errors in consumption, we must strongly reject the static UMT. Our findings call into question the robustness of the deterministic RP test that is due to Afriat (1967) and Varian (1982) to measurement error in prices.

6 See Afriat (1967), Varian (1990), and Echenique et al. (2011).

## Outline

The paper proceeds as follows. Section 2 presents the first-order conditions approach to the deterministic RP methodology. Section 3 contains our statistical test. Section 4 presents a framework for recoverability and counterfactual analysis on the basis of our testing methodology. Section 5 provides an econometric framework for our methodology. Section 6 provides a guide specifying the centering condition in different environments. Section 7 implements our empirical test for the case of the dynamic UMT in a consumer panel-survey data set. Section 8 implements our methodology for the case of the static UMT in an experimental data set. Section 9 presents a brief discussion of related literature. Finally, we conclude in Section 10. All proofs can be found in Appendix A.

## 2. The Revealed-Preference Methodology and the First-Order Conditions Approach

The main objective of this section is to provide a brief summary in a united fashion of two very important deterministic consumer models and their RP characterization. In particular, we study the static UMT or rational model (R), and the dynamic UMT with exponential discounting (ED). These models are at the center of many applied and theoretical works. We show that they can be completely characterized by their first-order conditions in an RP fashion. All quantities used here are assumed to be measured precisely.

Let the consumption space be R L + \{ 0 } , where L ∈ N is the number of commodities. 7 Consider a consumer who is endowed with a utility function u : R L + → R that is assumed to be concave, locally nonsatiated, and continuous. The consumer faces a sequence of decision problems indexed by t ∈ T , where T = { 0 , · · · , T } , with a known and finite T ∈ N . At each decision problem t ∈ T , the consumer faces the price vector p t ∈ R L ++ .

Definition 1 (Static UMT, R-rationalizability) . Adeterministic array ( p t , c t ) t ∈T is R-rationalizable (in a static sense) if there exists a concave, locally nonsatiated, and continuous function u , and some constants y t &gt; 0, t ∈ T , such that the consumption bundle c t solves:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all t ∈ T .

7 We use N to denote the set of natural numbers. The expression R L + denotes the set of componentwise nonnegative elements of the L -dimensional Euclidean space R L , and R L + \ { 0 } denotes the set of vectors v ∈ R L + that are distinct from zero ( v = 0). Similarly, R L ++ denotes the set of componentwise positive elements of R L + . The inner product of two vectors v 1 , v 2 ∈ R L is denoted by v ′ 1 v 2 .

/negationslash

Next we focus on the dynamic UMT. We assume that an individual consumer has preferences over a stream of dated consumption bundles ( c t ) t ∈T , where T = { 0 , · · · , T } , T ∈ N , and c t ∈ R L + \ { 0 } . (The number of goods, L , is kept the same across the time interval.) At time τ , the consumer chooses how much c τ she will consume by maximizing

<!-- formula-not-decoded -->

subject to the linear budget or flow constraints shown here:

<!-- formula-not-decoded -->

where d ∈ (0 , 1] is the discount factor; p t ∈ R L ++ is the price vector as before; y t ∈ R ++ is income received by the individual at time t ; s t is the amount of savings held by the consumer at the end of time t ; and a t is the volume of assets held at the start of time t . The consumer invests all her savings. Moreover, the assets evolve according to the following law of motion:

<!-- formula-not-decoded -->

where r t +1 &gt; -1 is the interest rate that is accessible for the consumer. The holdings of assets in the last period ( t = T ) are set to be zero.

The intertemporal value function, V t : R L × ( T -t +1) + → R ++ , represents the consumer preferences at a given time t . The components of this representation are the parameters of the model. First, d ∈ (0 , 1] is a scalar number that measures the degree of discount that the consumer gives to the future. Second, u : R L + → R ++ is an instantaneous utility function that is assumed to be concave, locally nonsatiated, and continuous. The exponential discounting consumer is time-consistent, that is, she will solve the dynamic problem above the same way at any point of the time window.

Definition 2 (Dynamic UMT, ED-rationalizability) . A deterministic array ( p t , r t , c t ) t ∈T is EDrationalizable if there exist a concave, locally nonsatiated, and continuous function u , a vector ( y t ) t ∈T ∈ R |T | ++ , and a scalar a 0 ≥ 0 such that the consumption stream ( c t ) t ∈T solves:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

subject to

ED-rationalizability implicitly assumes perfect foresight (e.g., individuals know their future income) and homogeneity of consumers within a household (e.g., household members have the same discount factors). In Appendix E we show that our methodology covers two extensions of this model: the dynamic UMT with income uncertainty, and the collective model of Adams et al.

(2014).

## 2.1. The First-Order Conditions Approach

Now we establish that any consumer model m ∈ { R , ED } can be completely characterized in terms of its first-order conditions with respect to (i) a concave, locally nonsatiated, and continuous utility function u : R L + → R , (ii) the effective (or transformed) prices ρ m t ∈ R L ++ , and (iii) restrictions on some constants λ m t ∈ R ++ and δ m t ∈ (0 , 1], interpreted as the marginal utility of income and the discount rate, respectively. We call this the first-order conditions approach . Observe that the utility function is model-independent, but the effective prices, the marginal utility of income, and the discount rate are not. We define the effective prices in Table 1.

| Table 1 - Definition of ρ m t      |
|------------------------------------|
| m R ED                             |
| ρ m t p t p t / ∏ t j =1 (1+ r j ) |

The following lemma summarizes the results in Browning (1989) for the exponential discounting case, and it is trivial for the static rationalizability case. Let ∇ u ( c t ) denote a supergradient of u at the point c t . 8

Lemma 1. For any model m ∈ { R , ED } , a deterministic array ( ρ m t , c t ) t ∈T is m -rationalizable if and only if there exists ( u, ( λ m t , δ m t ) t ∈T ) such that

- (i) u : R L + → R is a concave, locally nonsatiated, and continuous utility function;

/negationslash

- (ii) δ m t ∇ u ( c t ) ≤ λ m t ρ m t for every t ∈ T . If c t,j = 0 , then δ m t ∇ u ( c t ) j = λ m t ρ m t,j , where c t,j , ∇ u ( c t ) j , and ρ t,j are the j -th components of c t , ∇ u ( c t ) , and ρ t , respectively;
- (iii) λ R t = λ t &gt; 0 and δ R t = 1 for all t ∈ T ;
- (iv) λ ED t = 1 and δ ED t = d t , where d ∈ (0 , 1] , for all t ∈ T .

We want to highlight that while we focus on these two models for expositional and motivational purposes, our methodology is applicable to any model that can be characterized using the firstorder conditions approach.

Remark 1 . Lemma 1 allows for nondifferentiable utility functions. So, the supergradient of u ( c t ) may be set-valued. In this case one should read the condition δ m t ∇ u ( c t ) ≤ λ m t ρ m t as 'there exists ξ ∈ ∇ u ( c t ) such that δ m t ξ ≤ λ m t ρ m t . '

8 The supergradient is ∇ u ( c t ) = { ξ ∈ R L : u ( c ) -u ( c t ) ≤ ξ ′ ( c -c t ) , ∀ c ∈ R L + \ { 0 }} . Under differentiability, ∇ u ( c t ) is a gradient.

Remark 2 . Lemma 1 specialized for ED-rationality implies that we do not need to observe consumers over all periods of their lives. The first-order conditions are the same irrespective of whether the consumer lives any finite number of time periods containing the observed time-window, or is alive only during the latter period.

## 2.2. The Elimination of a Latent Infinite-Dimensional Parameter

Since our objective is not to estimate but to test m-rationalizability, we will eliminate the utility function u from its characterization. We follow the theorists of RP to eliminate the latent infinite-dimensional parameters by exploiting their shape restrictions.

In particular, we follow Afriat (1967), Varian (1985), Browning (1989), and Rockafellar (1970) to formulate a result that eliminates the utility function u (an infinite dimensional parameter) from the first-order conditions. The cost of doing this is that we have to replace the first-order conditions by a set of inequalities that require only the concavity of u . As a result, the inequalities are exact and do not involve any form of approximation; this is an advantage compared to other nonparametric methods (e.g., sieves, kernel estimators) or the parametric approach used in many applied papers.

To formulate our result, we first recall the definition of the concavity of u .

Definition 3 (Concavity) . A utility function u is said to be concave if and only if u (˜ c ) -u ( c ) ≤ ∇ u ( c ) ′ (˜ c -c ), for all c, ˜ c ∈ R L + \ { 0 } .

Remark 3 . In Definition 3 we implicitly assume the existence of the supergradient of u . Since the supergradient may be set-valued, one should read the condition u (˜ c ) -u ( c ) ≤ ∇ u ( c ) ′ (˜ c -c ) as ' u (˜ c ) -u ( c ) ≤ ξ ′ (˜ c -c ) for all ξ ∈ ∇ u ( c ). '

The nonparametric characterization of the m-rationalizability of observed consumption and prices without measurement error is captured by the following result.

Theorem 1. For any m ∈ { R , ED } , the following are equivalent:

- (i) The deterministic array ( ρ m t , c t ) t ∈T is m -rationalizable.
- (ii) There exist vectors ( λ m t ) t ∈T , ( δ t ) t ∈T , and a positive vector ( v t ) t ∈T such that:

<!-- formula-not-decoded -->

with λ R t = λ t &gt; 0 , δ R t = 1 , λ ED t = 1 , and δ ED t = d t , where d ∈ (0 , 1] , for all t, s ∈ T .

Theorem 1 summarizes known results from the RP literature. 9 Observe that Theorem 1 has transformed the first-order conditions that depend on the infinite-dimensional u to a set of inequalities that depend only on a deterministic finite-dimensional array ( v t , δ m t , λ m t ) t ∈T . Nonetheless, this

9 The proof is a consequence from the results in Afriat (1967), Varian (1985), and Browning (1989) taken together.

set of conditions is satisfied if and only if we can find a utility function that satisfies the conditions in Lemma 1. 10 Checking the set of inequalities is a parametric problem and it tells us whether a consumption stream is m-rationalizable. This methodology is traditionally applied at the individual level in panel data sets, assuming that the data contains no measurement error. In the next section we extend the RP framework to a noisy or stochastic environment.

## 3. The Revealed-Preference Approach with Measurement Error

In this section, we introduce a new statistical notion of m-rationalizability (henceforth, s / mrationalizability) with mismeasured consumption or prices, and provide a result similar to Theorem 1 in the presence of measurement error. From here on, we use boldface font to denote random objects and regular font for deterministic ones.

## 3.1. Statistical Rationalizability

We are interested in testing a statistical model of consumption such that each individual is an independent, identically distributed (i.i.d.) draw from some stochastic consumption rule. Note that by Lemma 1 the choice of a particular model m only affects the definition of the effective price, and the restrictions on the marginal utility of income and the discount rate. Henceforth, we fix some model such that the effective prices, and the restrictions on the marginal utility of income and the discount rate are known, and we omit the superscript m from the notation.

Using Lemma 1 as motivation, we directly define s / m-rationalizability as follows. Let ρ ∗ t ∈ P ∗ t ⊆ R L ++ and c ∗ t ∈ C ∗ t ⊆ R L + \ { 0 } denote random vectors of true effective prices and true consumption at time t , respectively. 11

Definition 4 (s / m-rationalizability) . A random array ( ρ ∗ t , c ∗ t ) t ∈T is s / m-rationalizable if there exists a tuple ( u , ( λ t , δ t ) t ∈T ) such that

- (i) u is a random, concave, locally nonsatiated, and continuous utility function;
- (ii) ( λt ) t ∈T is a positive random vector, interpreted as the marginal utility of income, supported on or inside a known set Λ ⊆ R |T | ++ ;

10 In contrast to Afriat's theorem for the static UMT, the assumption of concavity of the utility function is necessary in our framework. The reason is that concavity is testable in some cases that are different from static utility maximization (e.g., expected utility maximization, Polisson et al., 2020). Concavity guarantees that firstorder conditions are necessary and sufficient in a wide variety of models beyond the static UMT. For the additional generality our result requires this additional constraint.

11 For short, we use a . s . instead of 'almost surely. ' We denote (i) the probability of an event A by the expression P ( A ); (ii) the indicator function by 1 ( A ) = 1 when the statement A is true, otherwise it is zero; (iii) the mathematical expectation of any random vector z by the expression E [ z ]; (iv) the cardinality of a set A is given by the expression |A| ; and (v) the norm of a vector v is given by ‖ v ‖ .

- (iii) ( δ t ) t ∈T is a positive random vector, interpreted as time-varying discount factor, supported on or inside a known set ∆ ⊆ (0 , 1] |T | ;
- (iv) δ t ∇ u ( c ∗ t ) ≤ λtρ ∗ t a . s . for all t ∈ T ;
- (v) For every j = 1 , . . . , L and t ∈ T , it must be the case that P ( c ∗ t,j = 0 , δ t ∇ u ( c ∗ t ) j &lt; λtρ ∗ t,j ) = 0, where c ∗ t,j , ρ ∗ t,j , and ∇ u ( c ∗ t ) j denote the j -th components of c ∗ t , ρ ∗ t , and ∇ u ( c ∗ t ), respectively.

/negationslash

This definition means that for a given realization of (i) the utility function, (ii) the marginal utility of income, and (iii) the discount rate, the realized effective prices and the realized true consumption should fulfill the inequality δ t ∇ u ( c ∗ t ) ≤ λ t ρ ∗ t . This is a special case of the dynamic random utility model in which the preferences (captured by u ), the random discount factor (captured by ( δ t ) t ∈T ), and the distribution of the marginal utility of income (captured by ( λ t ) t ∈T ) are drawn at some initial time for each consumer, and then are kept fixed over time.

Several consumer models can be characterized by their first-order conditions and by restrictions on the marginal utility of income, as we observed in Section 2. For instance, we define the statistical version of R-rationalizability or s / R-rationalizability by requiring that the support of the marginal utility of income be strictly positive (i.e., Λ = R |T | ++ ), and the discount rate to be one (i.e., ∆ = { 1 } |T | ). Similarly, we define s / ED-rationalizability by imposing that Λ = { 1 } |T | , and the support ∆ be given by the restriction δ t = d t , where d is a random variable supported on (0 , 1]. The effective prices in each case are to be defined according to Table 1.

Given the definition of s / m-rationalizability, we can now formulate the stochastic version of Theorem 1.

Lemma 2. For a given random array ( ρ ∗ t , c ∗ t ) t ∈T , the following are equivalent:

- (i) The random array ( ρ ∗ t , c ∗ t ) t ∈T is s / m -rationalizable.
- (ii) There exist positive random vector ( v t ) t ∈T , ( λt ) t ∈T supported on or inside Λ , and ( δ t ) t ∈T supported on or inside ∆ such that

<!-- formula-not-decoded -->

Lemma 2 allows us to statistically test the s / m-rationalizability of ( ρ ∗ t , c ∗ t ) t ∈T . However, as the following example demonstrates, any test based on this notion of rationalizability cannot differentiate between 'almost' s / m-rationalizability and exact s / m-rationalizabilty (an issue first identified by Galichon &amp; Henry, 2013).

Example 1 (Hyperbolic Discounting) . Consider the case of a consumer who maximizes

<!-- formula-not-decoded -->

where β ∈ (0 , 1] is the present-bias parameter. It is easy to see that if β → 1, then the consumption stream generated by this model is arbitrarily close to the ED-rationalizable behavior.

Example 1 presents a random choice rule that is not s / m-rationalizable, but is arbitrarily close to being s / m-rationalizable. In other words, there may exist sequences of random arrays that are not s / m-rationalizable that converge to random arrays that are s / m-rationalizable. That is why we need to extend the notion of the consistency of a data set that is characterized by s / m-rationalizability.

Definition 5 (Approximate s / m-rationalizability) . We say that ( ρ ∗ t , c ∗ t ) t ∈T is approximately consistent with s / m-rationalizability if there exists a sequence of random variables ( v ′ j , λ ′ j , δ ′ j ) ′ ∈ R |T | + × Λ × ∆, j = 1 , 2 , . . . , such that

<!-- formula-not-decoded -->

for all s, t ∈ T .

## 3.2. Introducing Measurement Error

Theorem 1 and Lemma 2 provide testable implications of s / m-rationalizability. These implications depend solely on the distribution of λ = ( λ t ) t ∈T , δ = ( δ t ) t ∈T and v = ( v t ) t ∈T . The usual approach to testing s / m-rationalizability would amount to solving a (non)linear programming problem corresponding to Theorem 1 at the level of individual consumers. However, this common practice does not work any more in the presence of measurement error. When true consumption or true prices are measured erroneously, we observe not c ∗ t or ρ ∗ t but rather perturbed versions of them. (See Section 6 for a discussion of the reasons measurement error in consumption and prices arises in survey, experimental, and scanner data sets.)

Define the measurement error w = ( w t ) t ∈T ∈ W as the difference between reported consumption and prices, c = ( c t ) t ∈T and ρ = ( ρ t ) t ∈T ; and true consumption and prices, ( c ∗ t ) t ∈T and ( ρ ∗ t ) t ∈T . That is, where w c t = c t -c ∗ t and w p t = ρ t -ρ ∗ t for all t ∈ T .

<!-- formula-not-decoded -->

It is important to note that we define the measurement error. We do not make any assumptions about how the difference between observed and true quantities arises (i.e., we allow for measurement error to be multiplicative or additive). 12 Moreover, we do not need to assume that measurement error is independent of other variables, independent within time periods, or independent across goods.

By Lemma 2 we can immediately conclude that the observed x = ( ρ t , c t ) t ∈T can be s / mrationalized if and only if there exist ( λ t , δ t , v t , w t ) t ∈T , with ( λ t ) t ∈T supported on or inside Λ, and

12 Formally, this makes the support W depend on the support of both the observed and the true quantities. For simplicity we omit this dependency from the notation.

( δ t ) t ∈T supported on or inside ∆ such that

<!-- formula-not-decoded -->

However, we know that without restrictions on the distribution of measurement error, RP tests have no power. That is, there always exists a measurement error w such that the observed x is consistent with s / m-rationalizability. Hence, we require access to additional validation information about measurement error. The source of measurement error is different in different applications. That is why in this section we formulate a general restriction on the measurement error distribution that can be tailored for a given empirical application.

Recall that x ∈ X denotes observed quantities. Let e = ( λ ′ , δ ′ , v ′ , w ′ ) ′ ∈ E | X denote the vector of latent random variables, supported on or inside the conditional support E | X . We say that a mapping g M : X × E | X → R d M is a measurement error moment . We only require the following condition on measurement error.

Assumption 1 (Centered Measurement Error) . (i) The random vector e is supported on or inside the known support E | X . (ii) There exists a known measurement error moment g M : X × E | X → R d M such that

<!-- formula-not-decoded -->

The choice of function g M depends on the application and the assumptions the researcher is willing to make on the basis of the available knowledge about the nature of measurement error. In Section 6 we provide examples of moment conditions in data sets that are often used in the RP literature. The objects of interest for us are measurement error in consumption, expenditure, and prices.

## 4. Recoverability and Counterfactuals

Varian (1982, 1984) exploits the connections between empirical content and counterfactuals. In particular, Varian (1982) seems to be the first to think of nonparametric counterfactual analysis as specification testing. 13 Following these ideas, our formulation of rationalizability allows us to answer questions about the recoverability of, and counterfactual predictions for, different objects of interest.

In Section 4.1 we show how to recover different quantities of interest (e.g., average true consumption at a given t = τ ) from the s / m-rationalizable data set. In Section 4.2 we demonstrate how to make out-of-sample predictions for expected consumption in a way that is analogous to Blundell et al. (2014). In the presence of measurement error, distributional information about the

13 Recent work building on these connections includes Blundell et al. (2003) and Allen &amp; Rehbeck (2019) in demand analysis, and Norets &amp; Tang (2014) in the analysis of dynamic binary choice models.

primitives of the model of interest is inevitably lost. Hence, we cannot apply the traditional approach proposed by Varian (1982) to recover preferences and to do counterfactual analysis on an individual basis. Instead, we use this section to pose questions about the primitives of the model at the level of the population.

## 4.1. Recoverability

Assume that x = ( ρ t , c t ) t ∈T can be s / m-rationalized and Assumption 1 holds. Suppose that there is a finite-dimensional parameter of interest θ 0 ∈ Θ, where Θ is a compact subset of the Euclidean space. The parameter of interest is related to the model via the user-specified moment condition

<!-- formula-not-decoded -->

The function g R can take different forms depending on the different questions the user wants to answer. We provide some examples here.

Example 2 (Expected True Consumption and Expected True Consumption Change) . If θ 0 is the expected true consumption at t = τ , then g R ( x, e ; θ 0 ) = c τ -w c τ -θ 0 . If θ 0 is the expected difference in true consumption at t = τ +1 and t = τ , then g R ( x, e ; θ 0 ) = c τ +1 -w c τ +1 -c τ + w c τ -θ 0 .

The user may also be interested in testing the joint null hypothesis that (i) the consumer is s / ED-rationalizable and (ii) the random discount factor distribution has certain properties.

Example 3 (Average Random Discount Factor) . The user may be interested in testing whether the average value of the random discount factor is equal to a certain fixed value, in which case g R ( x, e ; θ 0 ) = d -θ 0 .

In addition, our framework allows us to have, as a special case, latent random variables with flexible support.

Example 4 (Support of the Random Discount Factor) . The user may be interested in whether the random time-discount factor d has a support on or inside [ θ 01 , θ 02 ] ⊆ (0 , 1]. Then, for θ 0 = ( θ 01 , θ 02 ) ′ , one can define g R ( x, e ; θ 0 ) = 1 ( θ 01 ≤ d ≤ θ 02 ) -1.

## 4.2. Counterfactual Out-of-Sample Predictions

We consider a counterfactual situation in which the user is given an out-of-sample effective random price vector ρ ∗ T +1 (supported in R L ++ ), a data set x = ( ρ t , c t ) t ∈T such that Assumption 1 holds, and she then asks two related questions. First, the user wants to know if there exists a counterfactual random consumption vector c ∗ T +1 such that the augmented random array { ( ρ ∗ t , c ∗ t ) t ∈T , ( ρ ∗ T +1 , c ∗ T +1 ) } is approximately s / m-rationalizable, where c ∗ t = c t -w c t and ρ ∗ t = ρ t -w p t .

Second, if the answer to the first question is affirmative, then the user will be interested in constructing confidence sets for some counterfactual finite-dimensional parameter θ 0 ∈ Θ. The parameter θ 0 satisfies the user-specified moment condition

<!-- formula-not-decoded -->

Both questions can be answered simultaneously with our characterization of s / m-rationalizability. Observe that the answer to the first question is negative if the random array ( ρ ∗ t , c ∗ t ) t ∈T is not s / mrationalizable. In contrast, if the random array ( ρ ∗ t , c ∗ t ) t ∈T is s / m-rationalizable, then the counterfactual exercise is equivalent to checking that the counterfactual price/consumption distribution is simultaneously compatible with s / m-rationalizability and the user-specified moment condition. Formally, to answer both questions, we define what it means for a random array ( ρ ∗ t , c ∗ t ) t ∈T to be counterfactually rationalizable (C / m-rationalizability) for a given ρ ∗ T +1 , θ 0 , and g C .

Definition 6 (C / m-rationalizability) . For a given ρ ∗ T +1 , g C , and θ 0 , a random array ( ρ ∗ t , c ∗ t ) t ∈T is approximately C / m-rationalizable if there exist c ∗ T +1 such that

- (i) The augmented random array { ( ρ ∗ t , c ∗ t ) t ∈T , ( ρ ∗ T +1 , c ∗ T +1 ) } is approximately s / m-rationalizable;

<!-- formula-not-decoded -->

Observe that if a random array ( ρ ∗ t , c ∗ t ) t ∈T is C / m-rationalizable for a given ( ρ ∗ T +1 , c ∗ T +1 ) and θ 0 , then it is also s / m-rationalizable. However, the opposite is not always true.

We can apply Lemma 2 to Definition 6 and get an extended system of the RP inequalities coupled with the counterfactual moment conditions g C . Moreover, we can define an identified set for counterfactual parameter values Θ 0 . Formally,

<!-- formula-not-decoded -->

We highlight that our framework can accommodate additional support restrictions on the counterfactual objects. A classical support constraint is a target out-of-sample expenditure level (i.e., ρ ∗′ T +1 c ∗ T +1 = 1 a . s . ) as in Varian (1982). We omit these constraints from our discussion to simplify exposition.

Example 5 (Average Varian Support Set) . We consider a moment

<!-- formula-not-decoded -->

with θ 0 ∈ Θ = R L + \{ 0 } as a hypothesized average-demand vector. Thus, Θ 0 is the Average Varian Support Set. Given ρ ∗ T +1 this set describes the bounds on the average demand that is compatible with the s / m-rationalizability of the random array ( ρ ∗ t , c ∗ t ) t ∈T .

Example 6 (Quantile Varian Support Set) . For s / R-rationalizability, we can consider the following moment condition:

<!-- formula-not-decoded -->

where θ = (¯ e c , φ ) ′ ∈ R ++ × [0 , 1] , ¯ e c is a fixed φ -quantile of the counterfactual expenditure distribution. Next we can define the φ -quantile Varian Support Set:

<!-- formula-not-decoded -->

This set describes the bounds of the counterfactual demand for a given ρ ∗ T +1 and φ -quantile of u ( c ∗ T +1 ) that is compatible with s / R -rationalizability.

Some counterfactual questions (e.g., the Average Varian Support Set) lead to convex identified sets Θ 0 .

Proposition 1. If the parameter space Θ is convex and g C is such that

<!-- formula-not-decoded -->

for some ˜ g and A , then Θ 0 is convex.

Proposition 1 imposes two restrictions on g C : (i) additive separability between c ∗ T +1 and θ ; (ii) affinity of the moment condition in θ . In Section 5 we provide a framework to construct confidence sets for the counterfactual parameters by means of the test inversion. Convexity of Θ 0 can substantially simplify the computation of the confidence sets since one does not need to conduct test inversion at every point of the parameter space.

## 5. Econometric Framework

In Sections 3.2 and 4.2 we showed how testing, recoverability, and counterfactuals in RP models with measurement error can be framed in the form of moment conditions. In this section we recast the empirical content of the RP inequalities in a form amenable to statistical testing. To simplify the exposition we will focus on testing s / m -rationalizability in the presence of measurement error ( g M and g I only).

## 5.1. Characterization of the Model via Moment Conditions

First, we write a set of moment conditions that will summarize the empirical content of s / m -rationalizability. Recall that x ∈ X denotes observed quantities and e = ( λ ′ , δ ′ , v ′ , w ′ ) ′ ∈ E | X denote the vector of latent random variables. The support E | X depends on the fixed supports Λ and ∆ that characterizes the particular model of interest. We use P X , P E,X , and P E | X to denote the set of all probability measures defined over the support of x , ( e ′ , x ′ ) ′ , and e | x , respectively. (Recall

that the boldface font letters denote random objects.) Define the following moment functions:

/negationslash

<!-- formula-not-decoded -->

We have k = |T | 2 -|T | and q = d M moment functions which correspond to inequality conditions ( g I ) and the measurement error centering conditions ( g M ), respectively. 14 Define E µ × π [ g ( x , e ) ] = ∫ X ∫ E | X g ( x, e ) dµdπ , where µ ∈ P E | X and π ∈ P X .

Theorem 2. The following are equivalent:

- (i) A random vector x = ( ρ t , c t ) t ∈T is approximately s / m -rationalizable such that Assumption 1 holds.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where π 0 ∈ P X is the observed distribution of x .

Theorem 2 establishes the equivalence between (i) s / m -rationalizability with the centered measurement error condition and (ii) a system of moment conditions. In other words, the observed consumption pattern, captured by the random array ( ρ t , c t ) t ∈T , can be s / m -rationalized under the restrictions on measurement error if and only if there exists a distribution of latent variables conditional on observables that satisfies the RP inequalities with probability 1, for the given supports Λ and ∆ .

Our notion of s / m -rationalizability makes clear that when one is dealing with measurement error, no RP test can decide whether a finite sample is consistent with model m . We can decide only that the data set is asymptotically consistent with the model as the sample size goes to infinity. Moreover, even asymptotically, there is no way to differentiate between the notion of approximate s / m -rationalizability and the notion of exact s / m -rationalizability. Nonetheless, we can do traditional hypothesis testing and decide at a fixed significance level whether we reject the null hypothesis of (approximate) model m consistency under Assumption 1 for a given sample. Conceptually, our notion of rationalizability corresponds to the extended notion of an identified set in Schennach (2014).

Note that the test is not yet formally established. We have a set of latent random variables e distributed according to an unknown µ ∈ P E | X . This problem can be solved nonparametrically using the Entropic Latent Variable Integration via Simulation (ELVIS) of Schennach (2014). The main advantage of the ELVIS approach is that it allows us to formulate a test that can be implemented in panel data sets suffering from measurement error of the type described only in terms of observables.

14 If in addition, the user includes moments g R or g C , then q = d M + d R or q = d M + d C , respectively.

## 5.2. ELVIS and Its Implications for Testing and Inference

We start this section by showing how the nonparametric results of Theorem 2 can be used to construct a set of (equivalent) parametric maximum-entropy moment conditions using Schennach (2014). Next, we provide a semi-analytic solution to the these conditions. Finally, we propose a procedure to test for s / m -rationalizability.

Following Schennach (2014), we define the maximum-entropy moment as follows.

Definition 7 (Maximum-Entropy Moment) . The maximum-entropy moment of the moment g ( x, · ) , for a fixed x , is where γ ∈ R k + q is a nuisance parameter, and η ∈ P E | X is an arbitrary user-input distribution function supported on E | X such that E π 0 [ log E η [ exp( γ ′ g ( x , e )) | x ] ] exists and is twice continuously differentiable in γ for all γ ∈ R k + q .

Note that is a family of exponential conditional probability measures. Thus, the maximum-entropy moment h is the marginal moment of the function g , at which the latent variable has been integrated out using one of the members from the above exponential family. The importance of the maximum-entropy moment is captured in the following result.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 3. The following are equivalent:

- (i) A random array x = ( ρ t , c t ) t ∈T is approximately s / m -rationalizable such that Assumption 1 holds.

(ii)

<!-- formula-not-decoded -->

where π 0 ∈ P X is the observed distribution of x .

The idea behind Theorem 3 is that if there exists a distribution that satisfies the moment condition, then there must be another distribution that (i) belongs to a particular finite-dimensional exponential family and (ii) satisfies the same moment condition. Since we are only interested in the existence of the former distribution, instead of searching over the set of all possible distributions we are going to only search over this 'smaller' exponential family.

We emphasize that Theorem 3 provides both necessary and sufficient conditions for the observed data to be (approximately) s / m -rationalizable. This represents an important gain in power with respect to any of the averaging-based tests of RP models that are usually used in the presence of measurement error.

High-level technical assumptions can ensure that the sequence of random latent variables that approximates model m converges to a proper random variable. Thus, this limiting random variable would ensure (i) that the infimum in Theorem 3 is attained, and (ii) that the notion of approximate rationalizability collapses to exact rationalizability. However, this obscures the fact that any assumption made in that direction has no testable implications.

The remarkable advantage of applying the results of Schennach (2014) to the RP approach is that it marginalizes out the latent random variables. More importantly, we have a robust statistical framework with which to test our models in the presence of measurement error. In particular, we have not made any strong distributional assumptions about λ , δ or u (the heterogeneous tastes). The only restrictions are the concavity assumption on the utility function, and a centering condition on measurement error. 15

Remark 4 . Theorem 3 does not imply that the distribution of the latent variables (or their support) is point-identified. In fact, it will always be set-identified.

Remark 5 . The maximum-entropy moment implicitly depends on the choice of the user-specified distribution η . However, η does not have any effects on the set of values E π 0 [ h ( x ; · ) ] or its sample analogue can take. That is, the choice of η will not affect the value inf γ ∈ R k + q ‖ E π 0 [ h ( x ; γ ) ] ‖ takes both asymptotically and in finite samples. The choice of η affects only the nuisance parameter ( γ ) value. See Remark 2.3 in Schennach (2014) for further details.

## 5.3. Semi-analytic Solution for the Maximum-entropy Moment

One can directly employ the maximum-entropy moment in Theorem 3 to test model m . However, doing so is potentially problematic. One possible concern is the fact that the number of maximum-entropy moments corresponding to the ( g I ) conditions, k = |T | 2 -|T | , grows quadratically with |T | . Moreover, γ 0 , the nuisance parameter value at which infimum is achieved, may be set-identified when unbounded (e.g., some of the components of γ 0 may be equal to infinity 16 ), which would therefore lead to nonstandard testing procedures.

Here we show that there exists a semi-analytic solution to the optimization problem where every component of γ 0 that corresponds to the RP inequality constraints is equal to + ∞ , and every component of γ 0 that corresponds to the measurement error centering constraint is finite and unique. Thus, for testing purposes (under the null hypothesis of model m ), we can minimize an objective function over a parameter space of lower dimensionality.

Assumption 2. (Nondegeneracy) There exist two subsets of E | X , E ′ and E ′′ , with a positive measure, such that componentwise the measurement error moment is such that sup e ∈ E ′ g M ( x , e ) &lt; 0 &lt; inf e ∈ E ′′ g M ( x , e ) with positive probability.

15 At this point, we can use as an alternative the methodology presented by Ekeland et al. (2010) to deal with latent variables in our moment conditions.

16 By 'equal to infinity' we mean that the infimum is achieved along the sequence of γ that diverges to infinity along some coordinates.

Assumption 3. (Bounded support) The random array x = ( ρ t , c t ) t ∈T has a bounded support.

Assumption 2 rules out cases in which there is no measurement error and allows us to a have a unique minimizer of the objective function. It can be relaxed since our methodology still works for cases without measurement error, but in those cases, it is preferable to use the equivalent deterministic RP benchmark. Assumption 3 is made to simplify the analysis and can be replaced by tail restrictions on the distribution of x .

Note that the user-specified distribution η should obey the same restrictions as the unknown distribution of latent e . Thus, we impose the following restrictions on η :

Definition 8 (User-specified distribution) . Almost surely in x , the user-specified distribution η ( ·| x ) satisfies all of the following:

- (i) The set ˜ E | X = { e ∈ E | X : g I ( x , e ) = 0 } has a positive measure under η ( ·| x ) .
- (ii) There exist two subsets of ˜ E | X , E ′ and E ′′ , with a positive measure under η ( ·| x ) , such that componentwise sup e ∈ E ′ g M ( x , e ) &lt; 0 &lt; inf e ∈ E ′′ g M ( x , e ) .
- (iii) For every finite γ M ∈ R q ,

<!-- formula-not-decoded -->

The first condition in Definition 8 requires that the support of η allows the inequalities to be satisfied. The second and third conditions are regularity conditions. This is a definition and not an assumption, as we can always construct an allowable η . 17 We are ready to present our main result.

Theorem 4. Given a user-specified measure η that satisfies the three conditions in Definition 8, the following are equivalent:

- (i) A random array x = ( ρ t , c t ) t ∈T is approximately consistent with s / m -rationalizability such that Assumptions 1, 2, and 3 hold.
- (ii) For any sequence { γ I,l } + ∞ l =1 that componentwise diverges to + ∞ ,

<!-- formula-not-decoded -->

The sequence of minimizers of (1), { γ M,l } , converges to some finite γ 0 ,M that does not depend on { γ I,l } + ∞ l =1 .

<!-- formula-not-decoded -->

17 Schennach (2014) provides a generic construction that can be used here.

<!-- formula-not-decoded -->

where

Moreover, the minimizer of (2) is finite, and is equal to γ 0 ,M .

<!-- formula-not-decoded -->

The intuition behind Theorem 4 is that the RP inequalities presented here restrict only the conditional support of the latent variables (including the measurement error). Hence, given the support restrictions captured by the RP inequalities, only the centering condition comes in the form of moments. 18

Theorem 4 substantially simplifies the conclusion of Theorem 3. First, we need to minimize the objective function over a much smaller parameter space ( R q instead of R k + q ). Thus, the problem becomes computationally tractable. Second, if the data is consistent with s / m -rationalizability, then the minimizer, γ 0 ,M , has to be finite and unique . Finally, to compute ˜ h M ( x, γ M ) in applications, one may need to use Markov Chain Monte Carlo (MCMC) methods by sampling from η . Theorem 4 implies that it suffices to sample from d ˜ η ( ·| x ) = 1 ( g I ( x, · ) = 0 ) dη ( ·| x ) . The straightforward way to sample from ˜ η is to sample from η and then reject the draw if it does not satisfy the RP inequalities captured by 1 ( g I ( x, · ) = 0 ) . The last part usually amounts to solving a linear program for R -rationalizability. For the case where this rejection sampling is not efficient, in Appendix C, we describe a new hit-and-run algorithm to sample from ˜ η ( ·| x ) directly. 19 This approach is particularly useful for testing s / ED -rationalizability in one of our applications.

## 5.4. Testing

Theorem 4 provides moment conditions that are necessary and sufficient for the data { x i } n i =1 = { ( ρ t,i , c t,i ) t ∈T } n i =1 (where n is the sample size), to be approximately consistent with s / m -rationalizability. Now, define the following sample analogues of the maximum-entropy moment and its variance:

<!-- formula-not-decoded -->

Let Ω -denote the generalized inverse of the matrix Ω . The testing procedure we propose is due to Schennach (2014) and is based on this test statistic:

<!-- formula-not-decoded -->

18 If Assumption 2 is violated for some component j of g M (e.g., sup E ′ g M,j ( x, e ) ≥ 0 a . s . for all E ′ ⊆ E | X ), then since E [ g M,j ( x , e )] = 0 it has to be the case that g M,j ( e , x ) = 0 a . s . . Thus, g M,j becomes a condition similar to g I that can be replaced by a support restriction.

19 The classical hit-and-run algorithm is an efficient MCMC method that generates uniform draws from a convex polytope. It initiates inside the polytope and proceeds by randomizing directions and then advancing a random distance while remaining inside the set.

Assumption 4. The data { x i } n i =1 is i.i.d.

Theorem 5. Suppose Assumptions 1, 2, 3, and 4 hold. Then under the null hypothesis that the data is approximately consistent with s / m -rationalizability, it follows that

<!-- formula-not-decoded -->

for every α ∈ (0 , 1) .

If, moreover, the minimal eigenvalue of the variance matrix V [ ˜ h M ( x , γ )] is uniformly, in γ , bounded away from zero and the maximal eigenvalue of V [ ˜ h M ( x , γ )] is uniformly, in γ , bounded from above, then, under the alternative hypothesis that the data is not approximately consistent with s / m -rationalizability, it follows that

<!-- formula-not-decoded -->

We conclude this section by noting that we can speed up computations of the test statistic by obtaining an initial guess for the minimizer of the objective function efficiently. We make use of the fact that, although the objective function ˆ ˜ h M ( · ) ′ ˆ ˜ Ω -( · ) ˆ ˜ h M ( · ) may have several local minima, the quadratic form ˆ ˜ h M ( · ) ′ B ˆ ˜ h M ( · ) , where B is any conformable positive definite matrix, has a unique global minimum and has no other local minima. 20

## 5.5. Confidence Sets for Parameters of Interest

The above testing procedure can be modified for construction of confidence sets for parameter θ 0 from Section 4.2. In particular, recall that one just needs to extend the set of the original moment conditions (the centering condition g M and the RP inequalities g I ) by g R or g C and add (if needed) extra RP inequalities that correspond to ( ρ ∗ T +1 , c ∗ T +1 ) . As in Section 5.4 we can then define TS n ( θ ) as the value of the test statistic computed for a fixed value of θ . Under assumptions similar to those of Theorem 5, the confidence set for θ 0 can be obtained by inverting TS n ( θ 0 ) . That is, the (1 -α ) -confidence set for θ 0 is

<!-- formula-not-decoded -->

where χ 2 q ext , 1 -α denotes the (1 -α ) quantile of the χ 2 distribution with q ext degrees of freedom ( χ 2 q ext ). The number q ext is determined by the number of the non RP moment conditions in the extended system that are not the support restrictions. Note that we do not pretest for s / m -rationalizability in order to construct the confidence set for θ 0 . If the data set is not s / m -rationalizable, then the confidence set will be empty asymptotically.

20 Additional details about the computational aspects of our methodology can be found in Appendix C.

## 6. Measurement Error in Different Data Sets

Recall that to apply our methodology we need measurement-error restrictions that come in the form of moments g M (Assumption 1). These moments capture the knowledge the user has about measurement error in a particular data set. In this section we provide examples of such centering conditions in different data sets relevant for the RP literature.

## 6.1. Measurement Error in Survey Data Sets

Measurement error in surveys may arise because of errors due to the respondent, the interviewer, or the survey design itself (Carroll et al., 2014). Recall mistakes, social desirability, and recording errors, among other factors, may cause measurement error in surveys (see Meyer et al., 2015). Household surveys usually measure expenditure across different goods, but do not measure consumption or prices directly. Prices are sometimes price indexes constructed by a national statistics agency. 21 Consumption is generated by dividing reported consumption expenditures in surveys by the observed prices aggregated to a category of goods (e.g., Blundell et al., 2003, Adams et al., 2014, and Kitamura &amp; Stoye, 2018). There is evidence that individuals, in either recall or self-reported surveys, can systematically overreport or underreport their expenditure on different categories of goods (Mathiowetz et al., 2002, Carroll et al., 2014, Pistaferri, 2015). In sum, both prices and consumption may have measurement error.

Consumption quantities are generated by dividing expenditures for a given good by its price index. This means that in the survey environment, measurement error in consumption is often nonclassical because it may combine measurement error of expenditures and prices in a nonlinear way (Attanasio et al., 2014). Nonetheless, we will argue that average total expenditures is wellmeasured in some surveys.

Abildgren et al. (2018) find that mean total expenditures are usually well-measured in the Danish interview-based household budgets survey (DHS, 2015). They use administrative registry information of the same households interviewed in the survey of interest to conclude that there are no statistically significant differences between the mean total expenditure in Danish households as reported in the registry data set and in the household survey. In addition, Kolsrud et al. (2017), using administrative registry consumption data from Sweden, also finds that mean household expenditures match those computed on the basis of household surveys. One potential caveat to this evidence is that registry data itself may be mismeasured. However, direct evidence on household expenditures from retail data shows that total household expenditure error in registry data has zero mean (Baker et al., 2018). 22

21 For instance, in data sets like the ones used in Blundell et al. (2003), Adams et al. (2014), and Kitamura &amp; Stoye (2018), price indexes are collected through direct observation in a given market and then merged with the expenditure survey data sets.

22 This pattern of measurement error in expenditure is also indirectly supported by evidence on measurement error in income. For instance, Angel et al. (2018), using data from Austria, show that the difference in household

Based on this evidence for measurement error, the analyst can impose the following condition. For all t ∈ T it must be the case that

<!-- formula-not-decoded -->

Equation (3) provides a measurement error moment that allows for nonclassical measurement error in consumption in the survey environment. In particular, it requires only that total expenditure measurement error be nonsystematic. Equation (3) implies that measurement error in consumption does not alter the mean value of total expenditures. In other words, it captures the idea that consumers, on average, may remember the total expenditure level better than the actual details.

Other surveys, such as the Spanish Continuous Expenditure Survey (1985-1997), used in Beatty &amp; Crawford (2011), Adams et al. (2014), and in one of our applications, have documented both over- and underreporting of income. Unfortunately, there is no validation data on expenditure itself (the object of our interest). However, Gradín et al. (2008) report that for the time period of interest the evolution of the mean household income and the mean household expenditure for Spain are very similar.

Spanish, Swedish, and Danish households self-report expenditures for a number of good categories in their respective household surveys. We see no reason to believe that Spanish subjects will behave differently from Danish ones, given the similar format of the surveys. For that reason, we believe it is reasonable to assume that the mean expenditure is usually well measured in the Spanish household survey.

However, in practice, the condition in Equation (3) may not be restrictive enough (i.e., the joint hypothesis of this assumption and s / m -rationality may not be rejected in general). We remind the reader that the measurement error moment is not the result of a modeling decision but should capture additional information that is usually obtained from validation exercises. Consequently, if we know too little about measurement error, then our data will not suffice to ensure a reasonable statistical power. A natural solution to this problem is to provide additional restrictions on measurement error by collecting new information about it. These additional restrictions will necessarily apply to specific settings or data sets.

A particular case of the condition captured by Equation (3), with the additional support constraint that prices are perfectly measured (i.e., ρ ∗ t = ρ t a . s . for all t ∈ T ), is:

Assumption 1.1. (Mean-Budget Neutrality for Survey Data Sets) For every t ∈ T ,

<!-- formula-not-decoded -->

Assumption 1.1 requires that measurement error in consumption be orthogonal to the effective prices. We maintain the assumption that prices are well measured in the rest of this subsection.

income reported from surveys and that from registry data displays mean-reverting behavior, as high income levels are underreported and low income levels are overreported. This evidence is consistent with the literature on measurement error in earnings (Bound et al., 2001).

We have three reasons for doing this. (i) This assumption is a common requirement in most of the work dedicated to the study of consumption decisions in households using survey data. In fact, we follow several works interested in modeling consumption in assuming that total expenditures and prices are measured better (i.e., with a higher signal-to-noise ratio) than the generated consumption variable. 23 (ii) It is empirically plausible that prices are better measured than consumption. In fact, there is evidence (Castillo et al., 1999, Guerrero de Lizardi, 2008) that the price index in Spain (which we use in one of our applications) is usually well measured with an estimated upward bias of the total price index of 0 . 07 percent for the time window 1981 -1991 (Castillo et al., 1999), which is arguably small. 24 For context, the estimated bias for the US in the well-known Boskin report was 0 . 40 percent for a similar time period (Boskin et al., 1998). In contrast, there is agreement that consumption measurement in surveys is very noisy (Alan et al., 2018). 25 (iii) We show that s / m -rationalizability is robust to measurement error when it is local (Appendix D.3). Hence, we will focus on the main source of measurement error that is consumption in the case of surveys. In sum, in the self-reported expenditures survey panel data sets, we argue that the main source of measurement error is in the consumption variable.

The mean-budget neutrality assumption is still compatible with nonclassical measurement error in consumption, such as w c l,t ≤ 0 a . s . ( w c l,t ≥ 0 a . s . ) for some good category l and all decision tasks t . That is, this condition allows agents on average to underreport (overreport) consumption on some categories of goods as long as they on average equally overreport (underreport) consumption on other categories of goods. Mean-budget neutrality will fail if measurement error in expenditures is systematic (i.e., every consumer overreports expenditures or underreports expenditures simultaneously).

We want to highlight the fact that despite its simplicity, Assumption 1.1 is a generalization of commonly held parametric assumptions about measurement error in the study of consumption. Here we provide some examples.

Example 7 (Multiplicative Measurement Error) . When one studies consumption of a single good across time (estimation of the Euler equation), one usually assumes that consumption measurement error is multiplicative (Alan et al., 2018). Formally, c t = c ∗ t /epsilon1 t , where /epsilon1 t can be assumed to be independent of, or at least orthogonal to, a set of instruments. For instance, we can assume that /epsilon1 t is independent of true consumption conditional on effective prices and with mean 1 . In that case, Assumption 1.1 holds. Alternatively, we can assume /epsilon1 t to be independent of effective prices and true consumption, which implies Assumption 1.1.

Example 8 (Additive Measurement Error) . Consider a case in which c t,l = c ∗ t,l + /epsilon1 t,l for l = 1 , . . . , L ; and where /epsilon1 t,l ∼ TN [ -a,a ] (0 , σ ) for l = 1 , . . . , L -1 (from a truncated normal with variance σ and

23 For examples of papers assuming that prices are perfectly measured and that measurement error is present only in consumption, see Cochrane (1991), Ventura (1994), Carroll (2001), Ludvigson &amp; Paxson (2001), Alan et al. (2009), Adams et al. (2014), Carroll et al. (2014), Toda &amp; Walsh (2015), and Alan et al. (2018).

24 The bias is defined as the difference between the Laspeyres price index as used in Spain and the ideal Fisher price index (Diewert, 1998). This is usually known as the substitution or aggregation bias.

25 For instance, Alan &amp; Browning (2010) estimate that roughly 80 percent of the variation in consumption growth rate is due to measurement error (this is for the PSID in the US).

bounds [ -a, a ] for some positive a &gt; 0 ) such that c t,l ≥ 0 a . s . . Note that this example is similar to the one used in Varian (1985) with constraints to impose the nonnegativity of consumption. Note that Assumption 1.1 holds because measurement error is independent of prices and mean zero.

We conclude this section by noting that even if prices are measured correctly, there is a potential source of error coming from the fact that price indexes are the result of aggregation of commodities into categories. This aggregation implies that price indexes do not exactly reflect actual prices faced by consumers. Thus, the budget sets faced by the consumers may be different from the ones implied by price indexes. 26 This problem is common to all demand analysis using survey data sets. Nonetheless, separability of the utility function or homotheticity within a category are examples of conditions under which commodity aggregation produces consumption and price indexes that are consistent with utility maximization (Jerison &amp; Jerison, 1994, Lewbel, 1996). If one is willing to impose such conditions, then a rejection of utility maximization, even in the presence of commodity aggregation error, means that rationality can be rejected at the disaggregated level. 27 In addition, a nonrejection of the UMT means that price indexes and consumption behave as if they are rational. That is, the substitution patterns observed in the data after commodity aggregation and their relation with relative changes in price indexes can be summarized by the first-order condition approach. 28

The aggregation error is not the only additional source of error in prices in survey data. For instance, since prices and expenditures are measured in different surveys, the observed prices in the first survey most likely are not those faced by some of the consumers in the second survey. 29 Further investigation of these additional sources of error are left for future research.

## 6.2. Measurement Error in Consumption or Prices in Experimental Data Sets

Measurement error in consumption or prices in experimental data sets arise due to difficulties in eliciting the intended choices of the experimental subjects. Indeed, the experimental elicitation of choice may be subject to random variation due to (i) the subject's misperception of some elements of the task, (ii) the level of understanding of the experimental design, and (iii) nonsystematic mistakes in implementing intended choice. In general, there is an imperfect relation between the elicited choice and the intended choice behavior that the experiment tries to measure (Gillen et al.,

26 We thank a referee for pointing this out.

27 Recent work by Sato (2020) provides nonparametric RP evidence in favor of weak separability and validity of several widely used price indexes for food and beverages categories using data from Japan. In addition, a survey by Shumway &amp; Davis (2001) finds evidence in favor of the conditions in Lewbel (1996).

28 Another possible interpretation of our analysis in survey environments is that we are testing for utilitymaximizing behavior in a budgeting problem. In the budgeting problem, the consumer allocates her income into different good categories to maximize her utility at that level of aggregation. Then, the consumer maximizes utility within different categories. This nested maximization problem is similar to mental accounting (Thaler, 1985) in its modern interpretation by Montgomery et al. (2019).

29 For instance, Gaddis (2016) documents that prices are sometimes collected by statistical offices in urban areas only, and that food prices can differ between urban and rural areas. However, price indexes constructed from these prices are matched with household expenditures surveys from both urban and rural areas.

2017).

We consider two sources of measurement error in the experimental environment in the context of the budget allocation task due to Choi et al. (2014) and Ahn et al. (2014) (which we use in our second application). The first one is the possibility of measurement error in consumption due to trembling-hand errors and the second one is measurement error in prices due to misperception.

We capture the relation between the true intended choices and the measured choices with the following assumption that allows for trembling-hand errors.

Assumption 1.2. (Trembling-Hand Errors for Experimental Data Sets) For all t ∈ T , it must be the case that

<!-- formula-not-decoded -->

Assumption 1.2 requires that measurement error in consumption be nonsystematic or equivalently centered around zero. Formally, for all t ∈ T it must be that

<!-- formula-not-decoded -->

We use data from Kurtz-David et al. (2019) to provide direct empirical evidence supporting Assumption 1.2. In a setup with two goods Kurtz-David et al. (2019) implement a motor task by asking their subjects to click on a visual target located on a budget line. Their experimental interface is exactly the same as in Choi et al. (2014). Kurtz-David et al. (2019) record the target coordinates and the actual coordinates that subjects choose on a screen using a mouse. The difference between the target coordinates and the actual choice is the trembling-hand error. Using this data set, we verified that the average trembling-hand error is statistically not different from zero. 30 We believe this is reasonable evidence in favor of Assumption 1.2 for budget allocation experiments.

Failure to account for the possibility of subjects' misperception of the experimental task may affect the elicitation of true consumer behavior. In particular, the experimental design of Choi et al. (2014) and Ahn et al. (2014) relies on a graphical representation of the budget hyperplane to elicit consumption choices. It is therefore a visual task and an economic task at the same time. We now consider the possibility of misperception of prices that can be thought of as measurement error in prices due to experimental design.

Assumption 1.3. (Misperception of Prices for Experimental Data Sets) For all t ∈ T , it must be the case that

<!-- formula-not-decoded -->

Assumption 1.3 relaxes the implicit requirement in the deterministic RP framework that subjects perceive the budget constraints without any distortion. Note that the graphical experimental device used by Choi et al. (2014) and Ahn et al. (2014) may make it difficult for consumers to correctly understand the true prices. We believe that it is desirable to have an RP test of the validity

30 Formally, we did a t-test of Assumption 1.2 across 23 subjects for 27 trials. The null hypothesis is not rejected at the 5 percent significance level for all except for one good at one trial, but the null hypothesis is not rejected in any instance at the 1 percent significance level. We did not perform a joint test because there are missing values.

of the UMT that is robust to the misperception of prices when the distortion is (i) attributable to the experimental design and (ii) nonsystematic.

We use data from Kurtz-David et al. (2019) to provide empirical evidence supporting Assumption 1.3. In contrast to the case of trembling-hand errors, there are no direct measurements on price misperception. However, Kurtz-David et al. (2019) collected data on visual misperception of coordinates that we can use to test Assumption 1.3 indirectly. In particular, we find evidence for a special case of this assumption that is compatible with the misperceived price being ρ l,t = ρ ∗ l,t / /epsilon1 l,t , for all goods l ∈ { 1 , · · · , L } , where /epsilon1 l,t captures misperception. Assumption 1.3 holds if the misperception error has mean 1 and is independent of true prices.

In a setup with two goods, Kurtz-David et al. (2019) implement a visual task by giving their subjects a particular numerical target z ∗ t = ( z ∗ 1 ,t , z ∗ 2 ,t ) . Next, the experimenters ask their subjects to locate this coordinate on a budget line. As a result, the point z t = ( z ∗ 1 ,t /epsilon1 1 ,t , z ∗ 2 ,t /epsilon1 2 ,t ) is observed, where we treat /epsilon1 t as the misperception error. Since prices can be computed from two observed points on the same budget line, misperceived prices will be ρ l,t = ρ ∗ l,t //epsilon1 l,t for l ∈ { 1 , 2 } . 31 A sufficient condition for Assumption 1.3 is that the multiplicative error is independent of true prices and that the mean perception error across individuals is 1 . We cannot check the independence condition from this validation data. Nonetheless, we cannot reject the null hypothesis that the mean (multiplicative) perception error is equal to 1 across the 25 subjects in this experiment, for each of the 27 trials (at the 5 percent significance level). 32

In the budget allocation tasks implemented by Choi et al. (2014) and Ahn et al. (2014), the subject is forced to choose a point on the budget line. Given that in these experimental environments the total income or wealth is known, the support of measurement error E | X must be such that ρ ∗′ t c ∗ t = ρ ′ t c t a . s . . 33 Note that in both the trembling-hand errors and in the misperception case we have d M = |T | · L . The number of centering conditions grows with the number of commodities and the number of decision tasks.

## 6.3. Measurement Error in Scanner Data Sets

Even though our main focus is on survey and experimental data sets, our methodology can be used in other types of data sets with their corresponding measurement error constraints. Here we provide a quick overview of the case of scanner data sets because of its relevance for RP practitioners.

Scanner consumer panel data sets are usually of high quality; thus, measurement error concerns may be less important. However, in some cases, like in the well-known Nielsen Homescan Scanner

31 Formally, we assume that the perception error realization is the same when the subject observes a second point given the same budget line.

32 We performed a t-test analogous to the case of the trembling-hand error.

33 In our application, we want to test separately for the validity of the static UMT together with (i) tremblinghand error in consumption or (ii) misperception of prices. Hence, we will assume that prices are perfectly measured in the former case (i.e., w p t = 0 a . s . ), while assuming that consumption is perfectly measured in the latter case (i.e, w c t = 0 a . s . ).

Data Set (NHS), there is evidence of measurement error in prices due to classical misreporting but also due to imputation done in the data collection stage (Einav et al., 2010). 34

Einav et al. (2010) use validation data from a retailer and compare it to a subsample of the self-reported NHS (for 2004 ); they conclude that consumption is measured rather precisely with roughly 90 percent of all records being exactly reported. On the other hand, prices are likely to be recorded with an upward bias. In fact, prices are measured precisely in only 50 percent of records in the sample of interest. The sample mean of the logarithm of prices 35 in the NHS seems to be slightly above the same quantity for the validation data. 36 Statistically, the difference between the logarithm of prices in the total sample of interest in the NHS and in the validation data is not different from zero (at the 5 percent significance level), as reported in Einav et al. (2010). 37

We believe that the findings of Einav et al. (2010) support the conclusion that consumption measurement error in the NHS can be treated as local perturbations (see Appendix D.3). Hence, we assume that consumption is measured precisely (i.e., w c t = 0 a . s . ) and impose the following centering condition for the NHS:

Assumption 1.4. (Centered Differences in Price Measurement Error) For all t, s ∈ T and all l = 1 , 2 , . . . , L , it must be the case that

<!-- formula-not-decoded -->

The above centering condition for measurement error allows for nonsystematic overreporting or underreporting of the logarithm of prices on average. Assumption 1.4 implies that the number of measurement error conditions is d M = |T | · L .

The source of measurement error in the NHS is very different from the one in survey data. In the NHS, households report quantities and prices, while in the survey data set, households report expenditures. For this reason survey data sets are usually used for aggregate analysis (at the level of the category of goods), in cases where price indexes computed by the national statistical agencies are available. As a result, the main mismeasured object is consumption. In contrast, scanner data sets have rich disaggregated information on prices and quantities. But misreported or imputed prices lead to measurement error. For that reason, we impose our centering condition on the main source of measurement error in each of these cases.

34 Similarly, in the older Stanford Basket Scanner Data Set (Echenique et al., 2014), there could be measurement error in prices due to unobserved coupons or discounts.

35 The mean is taken across all records in the time window of interest.

36 It seems that Nielsen generates the price in cases in which they have access to retailer-level price data. The reason for the overreporting is that this imputation process ignores the discounts that consumers may get (Einav et al., 2010).

37 In the subsample of records from the NHS in which the consumers did not get a sales discount, the measurement error in the logarithm of prices seems classical (i.e., centered around zero and symmetric). In contrast, in the subsample of records in which the consumers got a sales discount, the distribution of measurement error in the logarithm of prices has a fat right tail (Einav et al., 2010). As a result, the total measurement error in the logarithm of prices is not symmetric.

## 6.4. Other Forms of Measurement Error: Moment Inequalities and Instruments

Our methodology also allows for imposing moment inequality restrictions on measurement error. Following Schennach (2014), we can handle conditions of the type

<!-- formula-not-decoded -->

by introducing an additional slack positive random vector s = ( s j ) j ∈{ 1 , ··· ,d M } such that

<!-- formula-not-decoded -->

Moment inequalities may be particularly useful for taking into account bounds on measurementerror averages (e.g., E [ w p t ] ≤ 0 for all t ∈ T ). Imposing support constraints on measurement error (e.g., by rounding: | w l,t | ≤ 1 / 2 a . s . ) can be handled automatically by setting the support E | X appropriately.

Measurement error moments can also capture exclusion/orthogonality restrictions. In other words, the analyst may have information or be willing to assume that a particular observed variable is orthogonal to measurement error in consumption, prices or expenditures. The literature of demand estimation in both the static and dynamic setups, which use moments, usually handles measurement error through exclusion restrictions. 38

Note that Assumption 1.1 can be understood as an orthogonality restriction between prices and consumption measurement error (when prices are observed without error). Consider an instrumental variable z t supported on R L that is orthogonal to the consumption measurement error; this can be expressed as:

<!-- formula-not-decoded -->

for all t ∈ T . For Assumption 1.1, the variable z t = p t a . s . . These additional restrictions will increase the power of the test simply because they contain more information about measurement error.

## 6.5. Asymptotic Power for s/ED-Rationalizability: Illustrative Example

In previous sections we discussed different centering conditions that can be imposed on measurement error in different data sets. Since our characterization of s / m -rationality is necessary and sufficient, we have asymptotic power of one against the alternate hypothesis of inconsistency with s / m -rationalizability. Nonetheless, we still have to show that the alternative hypothesis space is nonempty. That is, we need to make sure that the restrictions on measurement error we provide are falsifiable. Here we build an illustrative example for s / ED -rationality and the centering condition captured by Assumption 1.1 with no price measurement error. Similar examples can be built for other moments used in our applications (see Appendix D). We also provide simulation evidence in

38 See Lewbel &amp; Pendakur (2009) and Alan et al. (2018).

Appendix B.2.

Consider the random array ( ρ t , c t ) t ∈T such that T = { 0 , 1 } , ρ 0 = (1 , 1) ′ , ρ 1 = (2 , 2) ′ , c 0 = (1 , 1) ′ , and c 1 = (2 , 2) ′ a . s . . This data set requires that prices and consumption be the same for all consumers in the population. Moreover, it is easy to see that in deterministic terms ED -rationality fails. We will now show that s / ED -rationality also fails with appropriate centering conditions. Assume towards a contradiction that this random array is s / ED -rationalizable. Then, there have to be random variables d ∈ (0 , 1] , { w c t } t =0 , 1 , and { v t } t =0 , 1 such that

<!-- formula-not-decoded -->

Combining these two inequalities, we get

<!-- formula-not-decoded -->

Thus, since 1 ≥ d &gt; 0a . s . , it follows that d ( v 1 -v 0 ) ≤ 0a . s . . However, this implies that v 0 ≥ v 1 a . s . . If Assumption 1.1 holds, then it must be the case that:

<!-- formula-not-decoded -->

As a result, applying the centering condition, we obtain a contradiction:

<!-- formula-not-decoded -->

This contradiction means that the random array ( ρ t , c t ) t ∈T is not s / ED -rationalizable under the centering conditions we described above.

We highlight that potential lack of statistical power of some centering conditions is not a defect of our methodology. The reason is that if the quality of the data set at hand is not good enough to credibly discern whether observed behavior is consistent with a given model, then no methodology can address this issue.

## 7. Empirical Application (I): Testing the Dynamic UMT with Exponential Discounting in Survey Data

In our first application, we apply our methodology to a consumer panel data set gathered from single-individual and couples' households in Spain to test for s / ED -rationalizability. This important model is under increasing scrutiny because experimental evidence tends to find that the

behavior of experimental subjects is time-inconsistent. 39 Nonetheless, it is important to explore to what extent this finding has external validity.

To address this issue, some researchers have turned to survey data in the form of household consumption panels. Most of this work has found evidence against exponential discounting (Blow et al., 2017). However, the existing literature has not yet addressed the issue of measurement error in the consumption reported by households in a way that allows us to perform traditional hypothesis testing. (Some additional problems with the existing evidence are (i) the strong parametric assumption on preferences, and (ii) homogeneity restrictions on the discount factor and preferences.)

One solution to some of the problems in the literature can be found in the work on deterministic RP by Browning (1989). In particular, Browning's work avoids making parametric assumptions about the functional form of instantaneous utility. However, this work does not take into consideration the fact that consumption can be mismeasured. In our Monte Carlo experiment, the deterministic test in Browning (1989) rejects the correct null hypothesis of exponential discounting behavior in 79 . 4 percent of the cases, while our methodology correctly fails to reject the null hypothesis that all households are consistent with exponential discounting at the correct 5 percent significance level (Appendix B.1). In addition, when we applied Browning (1989) deterministic methodology to our single-individual households data set, we also obtained a very low success rate. However, this low success rate of the deterministic test for exponential discounting may be due to measurement error. In our empirical application, we found support for exponential discounting behavior in single-individual households, while at the same time, support for the negative finding in Blow et al. (2017) in the case of couples' households. This fact indicates that deterministic tests may not be very informative about the behavior in a population due to measurement error. Small violations of the deterministic RP inequalities will lead to big rejection rates. Introducing measurement error into the analysis takes these small violations into account. 40

Our empirical application also contributes to the literature on estimating the discount factor distribution in survey data sets and in a classical consumer theory environment. This has been the topic of a large body of work which, however, has reached little or no consensus. 41 This lack of consensus can be attributed in some degree to a failure to identify the parameters of interest. Here, we show that the discount factor distribution cannot be identified solely from prices, interest rates, and consumption observations in a data set that suffers from measurement error. (For details see Section 4.) However, our methodology allows us to test for exponential discounting behavior even in this setting (i.e., without identifying the discount factor distribution). 42

If one ignores the issues of measurement error, the Euler equation allows one to estimate the

39 See, for instance, Andreoni &amp; Sprenger (2012), Montiel Olea &amp; Strzalecki (2014), and Echenique et al. (2014).

40 In Appendix E.2, we also establish that our test fails to rejects implications of the collective household consumption problem presented in Adams et al. (2014).

41 We refer the reader to the survey by Frederick et al. (2002) for its extensive references.

42 In order to learn more information about the discount factor distribution, one needs additional data. One notable example is Mastrobuoni &amp; Rivers (2016), which uses a quasi-experiment to pin down criminals' time preferences.

discount factor and the marginal utility either parametrically or semi-parametrically. 43 Since our objective is not to estimate but to test the exponential discounting model, we follow a different path.

In particular, we work with the data set used in Adams et al. (2014): the Spanish Continuous Family Expenditure Survey ( Encuesta Continua de Presupuestos Familiares ). The data set consists of the expenditures for 185 individuals and 2004 couples, as well as prices for 17 commodities (categories of goods) recorded over four time periods. Each household was interviewed for four consecutive quarters between 1985-1997. We construct a panel data set of consumption and prices by pooling household's quarterly data points.

The categories of goods are: all food and nonalcoholic drinks, all clothing, cleaning, nondurable articles, household services, domestic services, public transport, long-distance travel, other transport, petrol, leisure (four categories), other services (two categories), and food consumed outside the home. The data set also contains information on the nominal interest rate on consumer loans faced by the household in any particular quarter. 44

Formally, we test for s / ED -rationalizability with (i) effective prices equal to the discounted spot prices, ρ t = ρ ED t (defined in Table 1), (ii) random marginal utility of income equal to the discounted value of one unit of wealth, λ t = 1 a . s . , and (iii) δ t = d t where d is interpreted as the (time-invariant) random discount factor supported on or inside (0 , 1] . 45 Imposing the additional Assumptions 1.1, 2, and 4, we can apply our testing methodology developed in Section 5.4. Recall that Assumption 1.1 indicates that measurement error does not alter the mean value of total expenditure, E [ ρ ′ t c t ] = E [ ρ ′ t c ∗ t ] .

## 7.1. The Results

## Single-Individual Households

We apply the deterministic methodology of Browning (1989) to single-individual households. Our initial conclusion is that 84 . 3 percent of the single-individual households behave inconsistently with exponential discounting (even when allowing for substantially more heterogeneity than previous works). 46 Next, we revisit this conclusion using our methodology, which addresses measurement error, while allowing a heterogeneous discount factor. We find that we cannot reject exponential discounting. Formally, we find at the 5 percent significance level that the random ar-

43 Examples of estimators of the Euler equation and similar models include Hall (1978), Hansen &amp; Singleton (1982), Dunn &amp; Singleton (1986), Gallant &amp; Tauchen (1989), Chapman (1997), Campbell &amp; Cochrane (1999), Ai &amp; Chen (2003), Chen &amp; Ludvigson (2009), Darolles et al. (2011), Chen et al. (2014), and Escanciano et al. (2016).

44 We spare the reader more details and refer them instead to Adams et al. (2014) for further information on the data set.

45 When the discount factor is 0, it is easy to see that ED-rationality becomes equivalent to R-rationality. In practice, we pick the interval [0 . 1 , 1] as the largest possible support for discount factor. This interval contains most reasonable values in the literature (Frederick et al., 2002, Montiel Olea &amp; Strzalecki, 2014).

46 We search for each individual household discount factor d in the grid { 0 . 1 , 0 . 15 , · · · , 1 } . See Crawford (2010), Adams et al. (2014), and Blow et al. (2017) for discount factor ranges close to [0 . 9 , 1].

ray x = ( ρ t , c t ) t ∈T is s / ED -rationalizable with a random discount factor d supported on or inside [0 . 975 , 1] ( TS n = 6 . 480 , p-value = 0 . 166 ). We believe that the lower bound value of 0 . 975 , for the quarterly discount rate, is reasonable (it corresponds to a annualized discount rate of 0 . 9 ).

Although there is no agreement in the literature about what appropriate values for the discount factor are (Frederick et al., 2002), a common benchmark in applied work is to set the discount factor according to the real interest rate in the economy. 47 In our case, the lower bound of the quarterly discount factor is 0 . 975 . This discount factor bound corresponds to an annual real interest rate of 10 . 7 percent. This is roughly consistent with this benchmark for the average real interest rates observed in our sample. 48

However, at the 5 percent significance level, we cannot reject exponential discounting, when discounting is set at 1 a . s . ( TS n = 6 . 140 , p-value = 0 . 189 ). Of course, this does not mean that this is the value of the discount factor. In fact, our sample and our knowledge about measurement error are not enough to elicit the support of the distribution of discount factors. Nonetheless, we can differentiate between behavior consistent with exponential discounting and systematic departures from it (as seen in the power analysis of our method). In sum, our findings provide evidence supporting exponential discounting for singles under reasonable discount factors.

Using our methodology in data sets with more time periods or with additional information about measurement error will improve the information we can obtain about discount factors. However, we will see that identifying the support of the discount factor is not essential to provide informative bounds on average demand (see Appendix F). 49

## Couples' Households

For the couples' households, the deterministic test of Browning (1989) rejects the exponential discounting model for 89 . 8 percent of the observations. Although this number seems large, one should keep in mind that for single-individual households the same deterministic test rejects the model in 84 . 3 percent of the cases. At the same time, for d ∈ [0 . 1 , 1] , our method does not reject the exponential discounting model for single-individual households. But we do reject the model for couples' households. In the case of couples' households, the test statistic is TS n = 71 . 015 (p-value &lt; 10 -12 ). 50

47 DeJong &amp; Dave (2011) suggests setting the discount factor value to d = 1 / (1+ r ), where r is the average (across individuals) annual real interest rate.

48 Using the results from Section 4.1 we also tested several candidates for the average quarterly discount factor θ 0 = E [ d ]: 0 . 995 (TS n = 14 . 071, p-value = 0 . 015), 0 . 996 (TS n = 5 . 105, p-value = 0 . 403), and 0 . 997 (TS n = 2 . 967, p-value = 0 . 705). The smallest θ 0 that is not rejected at the 5 percent significance level, 0 . 996, corresponds to the annual average discount factor of 0 . 984 and the average annual real interest rate of 1 . 6 percent.

49 As robustness check we conducted the test for the discount factor supported on or inside [0 . 1 , 1], [0 . 5 , 1], and [0 . 9 , 1]. As expected, since all three intervals contain [0 . 975 , 1], the null hypothesis is not rejected. The test statistic TS n is equal to 6 . 476 with p-value = 0 . 166 for all three intervals.

50 As a robustness check, we also tested the model assuming that d is supported on or inside [0 . 5 , 1], [0 . 9 , 1], and d = 1 a . s . . As expected, since these intervals are contained in [0 . 1 , 1], the null hypothesis is rejected. The test statistic TS n (p-value) is equal to 71 . 015 ( &lt; 10 -12 ), 70 . 964 ( &lt; 10 -12 ), and 101 . 579 ( &lt; 10 -12 ), respectively.

## 7.2. Discussion and Related Work on Testing the Exponential Discounting Model

One possible concern about our methodology is that its power is low in survey data sets (due to a small T dimension of the data) given its nonparametric nature. However, in Appendix B.2 we report, for 1000 trials with a sample size of 2000 , a rejection rate greater than or equal to 72 percent (with a data generating process consistent with the collective model as in Adams et al., 2014).

Our results for couples' households provide the first nonparametric evidence which is robust to measurement error and which demonstrates that not all couples' households manifests behavior consistent with exponential discounting. In Appendix E.2, we establish that a suitable extension of our methodology fails to reject the collective household consumption problem presented in Adams et al. (2014). 51 This should convince practitioners about the importance of modeling intrahousehold decision-making when dealing with intertemporal choice. The rejection of exponential discounting behavior for couples' households can be better understood given theoretical results that show that aggregating time-consistent preferences may lead to time-inconsistent behavior (Jackson &amp; Yariv, 2015).

The deterministic methodology of Browning (1989) concludes that the fraction of households that is inconsistent with exponential discounting under the deterministic test is similar for both single individuals and couples, but our statistical test rejects in the latter case while reaching the opposite conclusion in the former case. The difference in conclusions is due to the fact that our test implicitly takes into account the severity of the violations of exponential discounting, and imposes the mean budget-neutrality assumption on the measurement error corrections. 52

Our main empirical finding is robust to price measurement error. Adding an additional source of measurement error would make the rationalizability notion less demanding. 53

## 8. Empirical Application (II): Static UMT in Experimental Data Sets with Trembling-Hand Errors and Misperception of Prices

In this section, we use our methodology to test the static UMT in the widely known experimental data set by Ahn et al. (2014). The experimental task consists of T = 50 independent decision trials with n = 154 subjects. Each decision is a portfolio problem. The subjects face three states of the world σ ∈ { 1 , 2 , 3 } . The subjects are given 100 tokens per task and they have to choose a bundle of Arrow securities, c t ∈ R 3 + , for a randomly drawn price vector p t ∈ R 3 + \ { 0 } . The

51 Mazzocco (2007) also provides evidence in favor of the collective model using a parametric methodology.

52 We have also tested s / ED-rationality under Assumption 1.2, which requires that consumption quantities measurement error is centered around zero as in Varian (1985). We strongly reject the null hypothesis in this case, providing an illustration of the importance of using empirically-backed centering conditions.

53 See Appendix D.3 for results on robustness of our methodology to local perturbations in prices.

subjects are forced to choose a bundle that satisfies Walras' law such that for every decision task it must be that p ′ t c t = 100 . The subjects receive a payment in tokens according to the probability of each state of the world at the end of each round. At the end of the experimental task one of the rounds was selected using a uniform distribution and the tokens payment corresponding to that round is paid in dollars. 54 The exchange rate is 0 . 05 dollars per token and the participation fee is 5 dollars.

This ingenious experimental device (due to Choi et al., 2014) has allowed the RP practitioners to collect a large number of observations per individual with high price variation. Beatty &amp; Crawford (2011) highlighted the importance of rich price variation to have enough power in the experimental design to detect violations of the UMT.

The deterministic RP test for the static UMT in this data set concludes that only 12 . 99 percent of the experimental subjects pass the test. At first sight, this is a striking result, because the majority of subjects seems to be inconsistent with the core consumer model in economics. We reexamine the robustness of this result to measurement error in consumption due to errors in the elicitation of the intended behavior of consumers. In our application, we found support for the static UMT in this experiment in stark contrast with the findings from the deterministic RP test.

Measurement error in the experimental environment may arise due to the nature of the design. Subjects are presented with a graphical representation of a 3 dimensional budget hyperplane, and they must choose the consumption bundle by pointing to a point in this hyperplane using a computer mouse or the arrow keys in a keyboard. We must note that there is a mechanical measurement error due to the resolution of the budget hyperplane which is 0 . 2 tokens. More important factors, such as a lack of expertise in the decision task, could lead the consumers to make implementation mistakes when trying to choose their preferred alternative. Kurtz-David et al. (2019) provide direct evidence of trembling-hand error in a budget allocation task similar to Ahn et al. (2014), as well as indirect evidence of visual misperception, as we have previously discussed. Nonetheless, the actual reason why the experimental design fails to elicit the intended decision task is not our main concern. We take the stand that a desirable test of the static UMT has to be robust to possible nonsystematic mistakes arising from any experimental design. We consider both trembling-hand errors in consumption and nonsystematic misperception of prices. Formally, we test for s / R -rationalizability with (i) effective prices equal to the prices at each trial ρ t = ρ R t (defined in Table 1), (ii) marginal utility of wealth ( λ t ) t ∈T supported on Λ = R |T | ++ , and (iii) random discount factor equal to 1 ( δ t = 1 a . s . for all t ∈ T ).

We note that there is evidence for trembling-hand errors and misperception errors in the budget allocation task we consider here (see Section 6). However, we fail to reject the null hypothesis of the static UMT when allowing only for price misperception. Evidently, it follows that we fail to reject the null hypothesis of the static UMT when allowing for both price misperception and trembling hand errors. In addition, we reject the null hypothesis of the static UMT when allowing only for trembling hand errors. Thus, we can conclude that the main source of error in this type of experimental tasks is most likely price misperception.

54 The subjects are informed that the probability of state σ = 2 is 1 / 3, and the joint probability of the states σ ∈ { 1 , 3 } is 2 / 3.

## Misperception of Prices

We consider the possibility of measurement error in prices arising from misperception. We investigate the case where consumers behave as if they are trying to maximize a utility function subject to a misperceived vector of prices. In this regard, we take the point-of-view of Gillen et al. (2017) that points out that misperception in a low stakes experimental environments may affect the external validity of the conclusions drawn from an experimental data set. Kurtz-David et al. (2019) provide indirect evidence of errors induced by the visual task in the budget allocation problem in the very closely related interface of Choi et al. (2014). We require that consumers' average perception of prices is unbiased, namely, for all t ∈ T it must be that

<!-- formula-not-decoded -->

This is captured in Assumption 1.3. In order to isolate the effect of misperception on the observed violations of the static UMT, we assume that consumer behavior is measured perfectly ( w c t = 0a . s . ). In addition, due to the experimental design in Ahn et al. (2014), it must be that true prices are such that p ∗′ t c ∗ t = 100 a . s . (i.e., Walras' law holds). The value of the test statistic is TS n = 17 . 879 (p-value &gt; 1 -10 -10 ). This is below the conservative critical value χ 2 150 , 0 . 95 = 179 . 581 . 55 We do not reject the null hypothesis of the static UMT in the presence of misperception of prices, when the average vector of prices is equivalent to the true vector of prices. 56 More importantly, this finding puts in perspective the rejection of the static UMT in experimental data sets that use the graphical representation of budget hyperplanes (Choi et al., 2014, Ahn et al., 2014). In particular, we find evidence that prices misperception matters. When we account for this possibility, we no longer reject the null hypothesis of the static UMT.

## Trembling-Hand Errors in Consumption

We say that measurement error in consumption is the result of a trembling-hand when it is centered at zero. This idea is captured in Assumption 1.2 that requires that for all t ∈ T it must be that E [ w c t ] = 0 . Also, we keep the restriction that the true prices and consumption satisfy Walras' law p ∗′ t c ∗ t = 100 a . s . , and that prices are measured perfectly w p t = 0 a . s . . As we discussed before, we have obtained direct evidence for this centering condition using the replication data of Kurtz-David et al. (2019). We strongly reject the null hypothesis of the static UMT when allowing for measurement error in the elicitation of the true consumer behavior due to trembling-hand errors. The test statistic is TS n = 299 . 137 (p-value &lt; 10 -11 ). This is above the conservative critical value χ 2 150 , 0 . 95 = 179 . 581 . 57

55 We used 900 draws in the Monte Carlo computation of the maximum-entropy moment of this problem. We chose this number on the basis of the trembling-hand error exercise in this experimental data set in the next section.

56 Assumption 1.3 has empirical bite. See Appendix D.2.

57 We used 2970 draws in the Monte Carlo computation of the maximum-entropy moment of this problem. We also tried 580 and 900 with test statistics with values of 301 . 654 and 306 . 882. Which is evidence that moderate size

## 9. Relation to the Literature

Afriat (1967) shows that it is sufficient to know shape constraints on the utility function (i.e., concavity) to bypass the need to know the utility function when testing for the UMT. We generalize this insight by allowing measurement error in consumption and prices. Among authors using the deterministic RP approach, the immediate antecedents to our work using the first-order approach are Browning (1989), Blow et al. (2017), and Brown &amp; Calsamiglia (2007). Important advances have been made on testing and doing counterfactual analysis under rationalizability or random utility. 58 However, the majority of these results assume that observed quantities are measured accurately. Varian (1985) is possibly the first work to introduce the subject of measurement error into the RP approach. Varian's methodology is the closest to that of our own work; he considers precisely measured (albeit random) prices to study measurement error in consumption. Varian's work is compatible with standard statistical hypothesis testing under the strong assumptions of normality (with known variance) and additivity of consumption measurement error. In contrast, our methodology is fully nonparametric. We are able to improve upon Varian's methodology and relax its core assumptions by using a moments approach to measurement error in the RP framework.

Other papers have dealt with measurement error under different parametric assumptions about measurement error or about the heterogeneity of preferences. 59 Deb et al. (2017) consider a nonparametric model of 'price preference.' They propose an RP test of their model that is robust to small measurement error in prices. Boccardi (2016) considers a case of demand with error and establishes a way to account for the trade-off between the fit of the model and its predictive ability (which is a generalization of Beatty &amp; Crawford, 2011). 60

In practice, the RP theorists (e.g., Adams et al., 2014 and Cherchye et al., 2017) have dealt with measurement error by perturbing (minimally) the observed individual consumption in order to satisfy the conditions of an RP test. For instance, Adams et al. (2014) find the additive perturbation with a minimal norm that renders the individual consumption streams compatible with the RP restrictions. Then a subjective threshold is imposed on the maximum admissible norm of the measurement error vector. If the computed norm is above the threshold, then the model is rejected. However, their methodology has one important drawback: every data set can be made to satisfy their test or, equivalently, the test has no power. In addition, the test in Adams et al.

of draws for the Monte Carlo integration step do relatively well in this setup.

58 Relevant examples are Blundell et al. (2014), Dette et al. (2016), Lewbel &amp; Pendakur (2017), and Kitamura &amp; Stoye (2018).

59 Gross (1995) assumes that random consumption is generated by consumers with similar preferences. Tsur (1989) imposes a log-normal multiplicative measurement-error structure in expenditures. Hjertstrand (2013) proposes a generalization of Varian (1985), but requires knowing the distribution of measurement error. Echenique et al. (2011) assume that measurement error in prices is a normal random variable independent across households and prices with a fixed mean and known variance.

60 The fit of an RP model indicates whether the a data set is consistent with the model. The predictive ability of an RP model is a measure of how easy it is for a data set generated at random to be consistent with the model. See Beatty &amp; Crawford (2011) for further details.

(2014) has no size control.

Among researchers using the RP approach, Blundell et al. (2003, 2014) are the first to provide consumer demand bounds, under the assumption of static utility maximization in a semiparametric environment (with additive heterogeneity) in which income changes continuously. Our work differs from theirs mainly in that we allow for unrestricted heterogeneity in preferences, do not require that income be observable, and do not impose semiparametric assumptions on wealth effects to provide bounds for demand, given new prices. In addition, nonclassical measurement error is not compatible with their approach.

## 10. Conclusion

We propose a new stochastic and nonparametric RP approach (suitable for an environment with measurement error in consumption or prices) that is useful to test for several consumer models that can be characterized by their first-order conditions. In particular, our work can be used (but is not limited) to test for static utility maximization (Afriat, 1967), and for dynamic rationalizability with exponential discounting (Browning, 1989).

## References

- Abildgren, K., Kuchler, A., Rasmussen, A. S. L., &amp; Sorensen, H. S. (2018). Consistency between household-level consumption data from registers and surveys. In 35th IARIW General Conference, Copenhagen, Danmarks Nationalbank Working Paper, no. 131 .
- Adams, A., Cherchye, L., De Rock, B., &amp; Verriest, E. (2014). Consume Now or Later? Time Inconsistency, Collective Choice, and Revealed Preference. American Economic Review , 104(12), 4147-4183.
- Afriat, S. N. (1967). The construction of utility functions from expenditure data. International economic review , 8(1), 67-77.
- Ahn, D., Choi, S., Gale, D., &amp; Kariv, S. (2014). Estimating ambiguity aversion in a portfolio choice experiment. Quantitative Economics , 5(2), 195-223.
- Ai, C. &amp; Chen, X. (2003). Efficient estimation of models with conditional moment restrictions containing unknown functions. Econometrica , 71(6), 1795-1843.

- Alan, S., Atalay, K., &amp; Crossley, T. F. (2018). Euler equation estimation on micro data. Macroeconomic Dynamics , (pp. 1-26).
- Alan, S., Attanasio, O., &amp; Browning, M. (2009). Estimating euler equations with noisy data: two exact gmm estimators. Journal of Applied Econometrics , 24(2), 309-324.
- Alan, S. &amp; Browning, M. (2010). Estimating intertemporal allocation parameters using synthetic residual estimation. The Review of Economic Studies , 77(4), 1231-1261.
- Allen, R. &amp; Rehbeck, J. (2019). Identification with additively separable heterogeneity. Econometrica , 87(3), 1021-1054.
- Andreoni, J. &amp; Sprenger, C. (2012). Estimating time preferences from convex budgets. The American Economic Review , 102(7), 3333-3356.
- Angel, S., Heuberger, R., &amp; Lamei, N. (2018). Differences between household income from surveys and registers and how these affect the poverty headcount: evidence from the austrian silc. Social indicators research , 138(2), 575-603.
- Attanasio, O., Hurst, E., &amp; Pistaferri, L. (2014). The evolution of income, consumption, and leisure inequality in the united states, 1980-2010. In Improving the measurement of consumer expenditures (pp. 100-140). University of Chicago Press.
- Baker, S. R., Kueng, L., Meyer, S., &amp; Pagel, M. (2018). Measurement error in imputed consumption . Technical report, National Bureau of Economic Research.
- Beatty, T. K. &amp; Crawford, I. A. (2011). How demanding is the revealed preference approach to demand? The American Economic Review , 101(6), 2782-2795.
- Blow, L., Browning, M., &amp; Crawford, I. (2013). Never Mind the Hyperbolics: Nonparametric Analysis of Time-Inconsistent Preferences. Unpublished manuscript .
- Blow, L., Browning, M., &amp; Crawford, I. (2017). Nonparametric analysis of time-inconsistent preferences. Mimeo.
- Blundell, R., Kristensen, D., &amp; Matzkin, R. (2014). Bounding quantile demand functions using revealed preference inequalities. Journal of Econometrics , 179(2), 112-127.
- Blundell, R. W., Browning, M., &amp; Crawford, I. A. (2003). Nonparametric engel curves and revealed preference. Econometrica , 71(1), 205-240.
- Boccardi, M. J. (2016). Predictive ability and the fit-power trade-off in theories of consumer behavior. Mimeo .
- Boelaert, J. (2014). revealedPrefs: Revealed Preferences and Microeconomic Rationality . R package version 0.2.

- Boskin, M. J., Dulberger, E. L., Gordon, R. J., Griliches, Z., &amp; Jorgenson, D. W. (1998). Consumer prices, the consumer price index, and the cost of living. Journal of economic perspectives , 12(1), 3-26.
- Bound, J., Brown, C., &amp; Mathiowetz, N. (2001). Chapter 59 - Measurement Error in Survey Data. In J. J. Heckman &amp; E. Leamer (Eds.), Handbook of Econometrics , volume 5 (pp. 3705-3843). Elsevier.
- Brown, D. J. &amp; Calsamiglia, C. (2007). The nonparametric approach to applied welfare analysis. Economic Theory , 31(1), 183-188.
- Browning, M. (1989). A nonparametric test of the life-cycle rational expections hypothesis. International Economic Review , (pp. 979-992).
- Browning, M., Chiappori, P.-A., &amp; Weiss, Y. (2010). Uncertainty and dynamics in the collective model.
- Campbell, J. Y. &amp; Cochrane, J. H. (1999). By force of habit: A consumption-based explanation of aggregate stock market behavior. Journal of political Economy , 107(2), 205-251.
- Carroll, C. D. (2001). Death to the log-linearized consumption euler equation!(and very poor health to the second-order approximation). Advances in Macroeconomics , 1(1).
- Carroll, C. D., Crossley, T. F., &amp; Sabelhaus, J. (2014). Introduction to" improving the measurement of consumer expenditures". In Improving the Measurement of Consumer Expenditures (pp. 1-20). University of Chicago Press.
- Castillo, J. R., Ley, E., &amp; Izquierdo, M. (1999). La medición de la inflación en España . Number 17. " la Caixa: Savings and Pensions Bank of Barcelona".
- Chapman, D. A. (1997). Approximating the asset pricing kernel. The Journal of Finance , 52(4), 1383-1410.
- Chen, X., Chernozhukov, V., Lee, S., &amp; Newey, W. K. (2014). Local identification of nonparametric and semiparametric models. Econometrica , 82(2), 785-809.
- Chen, X. &amp; Ludvigson, S. C. (2009). Land of addicts? an empirical investigation of habit-based asset pricing models. Journal of Applied Econometrics , 24(7), 1057-1093.
- Cherchye, L., Demuynck, T., De Rock, B., Vermeulen, F., et al. (2017). Household consumption when the marriage is stable. American Economic Review , 107(6), 1507-1534.
- Choi, S., Kariv, S., Müller, W., &amp; Silverman, D. (2014). Who is (more) rational? The American Economic Review , 104(6), 1518-1550.
- Cochrane, J. H. (1991). A simple test of consumption insurance. Journal of political economy , 99(5), 957-976.

- Crawford, I. (2010). Habits revealed. The Review of Economic Studies , 77(4), 1382-1402.
- Darolles, S., Fan, Y., Florens, J.-P., &amp; Renault, E. (2011). Nonparametric instrumental regression. Econometrica , 79(5), 1541-1565.
- Deb, R., Kitamura, Y., Quah, J. K.-H., &amp; Stoye, J. (2017). Revealed price preference: Theory and stochastic testing. Working paper .
- DeJong, D. N. &amp; Dave, C. (2011). Structural macroeconometrics . Princeton University Press.
- DellaVigna, S. &amp; Malmendier, U. (2006). Paying not to go to the gym. The American Economic Review , (pp. 694-719).
- Dette, H., Hoderlein, S., &amp; Neumeyer, N. (2016). Testing multivariate economic restrictions using quantiles: the example of slutsky negative semidefiniteness. Journal of Econometrics , 191(1), 129-144.
- Diewert, W. E. (1998). Index number issues in the consumer price index. Journal of Economic Perspectives , 12(1), 47-58.
- Diewert, W. E. (2012). Afriat's theorem and some extensions to choice under uncertainty. The Economic Journal , 122(560), 305-331.
- Dunn, K. B. &amp; Singleton, K. J. (1986). Modeling the term structure of interest rates under non-separable utility and durability of goods. Journal of Financial Economics , 17(1), 27-55.
- Echenique, F., Imai, T., &amp; Saito, K. (2014). Testable implications of quasi-hyperbolic and exponential time discounting.
- Echenique, F., Lee, S., &amp; Shum, M. (2011). The money pump as a measure of revealed preference violations. Journal of Political Economy , 119(6), 1201-1223.
- Einav, L., Leibtag, E., &amp; Nevo, A. (2010). Recording discrepancies in nielsen homescan data: Are they present and do they matter? QME , 8(2), 207-239.
- Ekeland, I., Galichon, A., &amp; Henry, M. (2010). Optimal transportation and the falsifiability of incompletely specified economic models. Economic Theory , 42(2), 355-374.
- Escanciano, J. C., Hoderlein, S., Lewbel, A., Linton, O., &amp; Srisuma, S. (2016). Nonparametric euler equation identification and estimation. Working Paper .
- Forges, F. &amp; Minelli, E. (2009). Afriat's theorem for general budget sets. Journal of Economic Theory , 144(1), 135-145.
- Frederick, S., Loewenstein, G., &amp; O'Donoghue, T. (2002). Time Discounting and Time Preference: A Critical Review. Journal of Economic Literature , 40(2), 351-401.
- Gaddis, I. (2016). Prices for Poverty Analysis in Africa . The World Bank.

Galichon, A. &amp; Henry, M. (2013). Dilation bootstrap. Journal of Econometrics , 177(1), 109 - 115.

- Gallant, A. R. &amp; Tauchen, G. (1989). Seminonparametric estimation of conditionally constrained heterogeneous processes: Asset pricing applications. Econometrica: Journal of the Econometric Society , (pp. 1091-1120).
- Gauthier, C. (2018). Nonparametric identification of discount factors under partial efficiency. Working paper .
- Gillen, B., Snowberg, E., &amp; Yariv, L. (2017). Experimenting with measurement error: techniques with applications to the Caltech cohort study . Technical report, National Bureau of Economic Research.
- Gradín, C., Cantó, O., &amp; Del Río, C. (2008). Inequality, poverty and mobility: Choosing income or consumption as welfare indicators. Investigaciones Económicas , 32(2), 169-200.
- Gross, J. (1995). Testing data for consistency with revealed preference. The Review of Economics and Statistics , (pp. 701-710).
- Guerrero de Lizardi, C. (2008). Sesgos de medición del índice nacional de precios al consumidor, 2002-2007. Investigación económica , 67(266), 37-65.
- Hall, R. E. (1978). Stochastic implications of the life cycle-permanent income hypothesis: theory and evidence. Journal of political economy , 86(6), 971-987.
- Hansen, L. P. &amp; Singleton, K. J. (1982). Generalized instrumental variables estimation of nonlinear rational expectations models. Econometrica: Journal of the Econometric Society , (pp. 12691286).
- Hjertstrand, P. (2013). A simple method to account for measurement errors in revealed preference tests. IFN Working Paper .
- Jackson, M. O. &amp; Yariv, L. (2015). Collective dynamic choice: the necessity of time inconsistency. American Economic Journal: Microeconomics , 7(4), 150-178.
- Jerison, D. &amp; Jerison, M. (1994). Commodity aggregation and slutsky asymmetry. In Models and Measurement of Welfare and Inequality (pp. 752-764). Springer.
- Kitamura, Y. &amp; Stoye, J. (2018). Nonparametric analysis of random utility models. Econometrica , 86(6), 1883-1909.
- Kolsrud, J., Landais, C., &amp; Spinnewijn, J. (2017). Studying consumption patterns using registry data: lessons from swedish administrative data.
- Kurtz-David, V., Persitz, D., Webb, R., &amp; Levy, D. J. (2019). The neural computation of inconsistent choice behavior. Nature communications , 10(1), 1583.

- Lewbel, A. (1996). Aggregation without separability: A generalized composite commodity theorem. The American Economic Review , 86(3), 524-543.
- Lewbel, A. &amp; Pendakur, K. (2009). Tricks with hicks: The easi demand system. American Economic Review , 99(3), 827-63.
- Lewbel, A. &amp; Pendakur, K. (2017). Unobserved preference heterogeneity in demand using generalized random coefficients. Journal of Political Economy , 125(4), 1100-1148.
- Ludvigson, S. &amp; Paxson, C. H. (2001). Approximation bias in linearized euler equations. Review of Economics and Statistics , 83(2), 242-256.
- Mastrobuoni, G. &amp; Rivers, D. (2016). Criminal discount factors and deterrence. Available at SSRN 2730969 .
- Mathiowetz, N., Brown, C., &amp; Bound, J. (2002). Measurement error in surveys of the low-income population. Studies of welfare populations: Data collection and research issues , (pp. 157-194).
- Mazzocco, M. (2007). Household Intertemporal Behaviour: A Collective Characterization and a Test of Commitment. The Review of Economic Studies , 74(3), 857-895.
- Meyer, B. D., Mok, W. K., &amp; Sullivan, J. X. (2015). Household surveys in crisis. Journal of Economic Perspectives , 29(4), 199-226.
- Montgomery, A., Olivola, C. Y., &amp; Pretnar, N. (2019). A structural model of mental accounting. Available at SSRN 3472156 .
- Montiel Olea, J. L. &amp; Strzalecki, T. (2014). Axiomatization and Measurement of Quasi-hyperbolic Discounting. Quarterly Journal of Economics , 129, 1449-1499.
- Norets, A. &amp; Tang, X. (2014). Semiparametric inference in dynamic binary choice models. Review of Economic Studies , 81(3), 1229-1262.
- Pistaferri, L. (2015). Household consumption: Research questions, measurement issues, and data collection strategies. Journal of Economic and Social Measurement , 40(1-4), 123-149.
- Polisson, M., Quah, J. K.-H., &amp; Renou, L. (2020). Revealed preferences over risk and uncertainty. American Economic Review , 110(6), 1782-1820.
- Powell, M. J. (2009). The bobyqa algorithm for bound constrained optimization without derivatives. Cambridge NA Report NA2009/06, University of Cambridge, Cambridge , (pp. 26-46).
- Rios, L. M. &amp; Sahinidis, N. V. (2013). Derivative-free optimization: a review of algorithms and comparison of software implementations. Journal of Global Optimization , 56(3), 1247-1293.
- Rockafellar, R. T. (1970). Convex analysis . Princeton university press.

- Sasaki, Y. (2015). A contraction fixed point method for infinite mixture models and direct counterfactual analysis.
- Sato, H. (2020). Do large-scale point-of-sale data satisfy the generalized axiom of revealed preference in aggregation using representative price indexes?: A case involving processed food and beverages. Mimeo .
- Schennach, S. M. (2014). Entropic latent variable integration via simulation. Econometrica , 82(1), 345-385.
- Shumway, C. R. &amp; Davis, G. C. (2001). Does consistent aggregation really matter? Australian Journal of Agricultural and Resource Economics , 45(2), 161-194.
- Thaler, R. (1985). Mental accounting and consumer choice. Marketing science , 4(3), 199-214.
- Toda, A. A. &amp; Walsh, K. (2015). The double power law in consumption and implications for testing euler equations. Journal of Political Economy , 123(5), 1177-1200.
- Tsur, Y. (1989). On testing for revealed preference conditions. Economics Letters , 31(4), 359-362.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica: Journal of the Econometric Society , (pp. 945-973).
- Varian, H. R. (1984). The nonparametric approach to production analysis. Econometrica: Journal of the Econometric Society , (pp. 579-597).
- Varian, H. R. (1985). Non-parametric analysis of optimizing behavior with measurement error. Journal of Econometrics , 30(1-2), 445-458.
- Varian, H. R. (1990). Goodness-of-fit in optimizing models. Journal of Econometrics , 46(1-2), 125-140.
- Ventura, E. (1994). A note on measurement error and euler equations: An alternative to log-linear approximations. Economics Letters , 45(3), 305-308.

## A. Appendix

## A.1. Proof of Lemma 2

First we establish that (i) implies (ii). If the random array ( ρ ∗ t , c ∗ t ) t ∈T is s / m -rationalizable, by concavity of u ( · ) , with probability 1 for any s, t and some ξ ∈ ∇ u ( c ∗ t )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let N be a random set of indices such that λ t δ t ρ ∗ ti = ξ i for every i ∈ N . Hence, λ t δ t ρ ∗ ti ≥ ξ i for every i /negationslash∈ N with probability 1 . As a result, c ∗ ti has to be a corner solution for every i /negationslash∈ N . That is, c ∗ ti = 0 . Thus, with probability 1 ,

<!-- formula-not-decoded -->

where the last inequality follows from c s being nonnegative. As a result, with probability 1 ,

<!-- formula-not-decoded -->

For any /epsilon1 &gt; 0 , we let v t = u ( c ∗ t ) -min s ∈T u ( c ∗ s ) + /epsilon1 a . s . , for all t ∈ T . The well-defined positive random vector ( v t ) t ∈T together with ( λ t , δ t ) t ∈T satisfies the inequalities in (ii).

Now, we want to prove that (ii) implies (i). The result follows from Theorem 24.8 in Rockafellar (1970). For completeness of the proof we repeat the arguments of Theorem 24.8 in Rockafellar (1970). For a finite m ∈ N , let t = { t i } m i =1 , t i ∈ T , be a finite set of indices such that for a fixed ˆ t ∈ T , c ∗ t 1 = c ∗ ˆ t . Let I be the collection of all such indices (i.e., t ∈ I ). Define

<!-- formula-not-decoded -->

With probability 1 , the random function u ( · ) is well-defined, concave, locally nonsatiated, and continuous, since it is a pointwise minimum of a finite set of affine functions for every m . Moreover, the infimum in I is attained and achieved at a set of indices without repetitions (this is a consequence of (ii)). Indeed, under (ii), for any finite m , { t i } m i =1 and c ∗ s , s ∈ T , with probability 1 ,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

with probability 1 . (In particular, u ( c ∗ ˆ t ) = 0 .)

It is left to show that, with probability 1 , λ t δ t ρ ∗ t ∈ ∇ u ( c ∗ t ) for all t ∈ T . Fix some t ∈ T and δ &gt; 0 . By the definition of u ( · ) , there exists some { t i } m i =1 such that, with probability 1, u ( c ∗ t ) + δ &gt; λ t 1 δ t 1 ρ ∗′ t 1 ( c ∗ t 2 -c ∗ t 1 ) + · · · + λ t m δ t m ρ ∗′ t m ( c ∗ t -c ∗ t m ) ≥ u ( c ∗ t ) . Again, by the definition of u ( · ) , for

any c ∗

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the choice of δ , t and c ∗ was arbitrary, λ t δ t ρ ∗ t ∈ ∇ u ( c ∗ t ) for all t ∈ T .

## A.2. Proof of Proposition 1

Take any θ 1 ∈ Θ 0 , θ 2 ∈ Θ 0 , and λ ∈ [0 , 1] (if Θ 0 is empty, then the conclusion of the proposition follows trivially). Since θ i ∈ Θ 0 , i = 1 , 2 , by Theorems 2 and 3 there exist { µ i,k } ∞ k =1 , i = 1 , 2 , such that for both i = 1 , 2 . Consider θ λ = λθ 1 +(1 -λ ) θ 2 and µ λ,k = λµ 1 ,k +(1 -λ ) µ 2 ,k . Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by the triangular inequality,

<!-- formula-not-decoded -->

Thus, since g I,O and g M do not depend on θ , by the triangular inequality,

<!-- formula-not-decoded -->

The later means that θ λ ∈ Θ 0 . The fact that the choice of θ 1 , θ 2 , and λ was arbitrary implies that Θ 0 is convex.

## A.3. Proof of Theorem 3

The result is a direct application of Theorem 2, and Theorem 2.1 in Schennach (2014). For completeness of the proof we present Theorem 2.1 in Schennach (2014) using our notation below.

<!-- formula-not-decoded -->

Theorem (Theorem 2.1, Schennach, 2014) . Assume that the marginal distribution of x is supported on some set X ⊆ R d x , while the distribution of e conditional on x = x is supported on or inside the set E ⊆ R d e for any x ∈ X . Let h , g and η satisfy Definition 7. Then

<!-- formula-not-decoded -->

where π 0 ∈ P X is the observed distribution of x .

## A.4. Proof of Theorem 4

Recall that the first k = |T | 2 -|T | moments correspond to the inequality conditions, and the last q moments correspond to the measurement error centering conditions. Let γ I = ( γ j ) j =1 ,...,k , g I = ( g j ) j =1 ,...,k , γ M = ( γ j ) j = k +1 ,...,k + q , and g M = ( g j ) j = k +1 ,...,k + q be sub-vectors of γ and g that correspond to inequality and the measurement error centering conditions, respectively.

Step 1. Take a sequence { γ I,l } + ∞ l =1 such that every component of γ I,l diverges to + ∞ . Note that since g I takes values in {-1 , 0 } k ,

<!-- formula-not-decoded -->

where γ I,l,i is the i -th component of γ I,l . Hence, for any function f ∈ L 1 ( η ( ·| x ))

<!-- formula-not-decoded -->

Hence, the sequence of measures exp( γ ′ I,l g I ( x, · )) dη ( ·| x ) converges to the measure

<!-- formula-not-decoded -->

in total variation. The later measure is well-defined and nontrivial since we assume that ˜ E | X = { e : 1 ( g I ( x, e ) = 0 ) } has a positive measure under η ( ·| x ) . Let d ˜ η ( ·| x ) denote 1 ( g I ( x, · ) = 0 ) dη ( ·| x ) . Step 2. Consider the moment conditions under d ˜ η ( ·| x )

<!-- formula-not-decoded -->

Definition 8.(iii) together with Assumption 3 and Step 1 imply that for any compact set Γ ∈ R q , uniformly in γ M ∈ Γ

<!-- formula-not-decoded -->

Thus, by continuity of h M in γ M , when l goes to infinity, we can work with the reduced optimization

problem:

Step 3. Note that (4) is equivalent to the optimization problem in Theorem 3. Hence, infimum in (4) is equal to 0 if and only if the data is approximately consistent with model m .

<!-- formula-not-decoded -->

We assumed that every component of g M takes both positive and negative values on some nonzero measure subsets of ˜ E | X (Assumption 2). Hence, following the proof of Theorem 2.1 and Lemma A.1 in Schennach (2014), we can conclude that if infimum in (4) is equal to 0 , then it is achieved at some finite and unique γ 0 ,M . Otherwise, ‖ γ M ‖ diverges to infinity.

## A.5. Proof of Theorem 5

The result is a direct application of Theorem F.1 in Schennach (2014). For completeness of the proof we present the version of it that is applicable to our setting below.

Theorem (Theorem F.1, Schennach, 2014) . Let data be i.i.d.. If (i) the set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is nonempty for some C &lt; ∞ ; (ii) E [ ∥ ∥ ∥ ˜ h M ( x , γ ) ∥ ∥ ∥ 2 ] &lt; ∞ for all γ ∈ Γ , then

An i.i.d. sample is assumed. To show the validity of conditions (i) and (ii) note that since x has a bounded support (by Assumption 3) and ˜ η satisfies conditions of Definition 8.(iii), for any finite γ there exist finite positive constant C 1 ( γ ) such that almost surely in x

<!-- formula-not-decoded -->

∥ ∥ Under the alternative hypothesis, ∥ ∥ ∥ ∥ ˆ ˜ h M ( γ ) ∥ ∥ ∥ ∥ either converges to a positive constant or diverges to infinity. Thus, since eigenvalues of ˜ Ω( γ ) are bounded away from zero and are bounded from above the test is consistent.

Hence, for any nonempty compact set Γ one can take C = sup γ ∈ Γ C 1 ( γ ) . Together with Assumption 3, the later implies condition (ii). Similarly, one can use C to bound E [∥ ∥ ˜ h M ( x , γ ) ∥ ∥ ] .

## A.6. Proof of Theorem 7

By Theorem 6 we have that the following inequalities hold almost surely:

<!-- formula-not-decoded -->

Then we multiply the first inequality by d t A , this random variable is positive almost surely, so it does not alter the inequalities. We do the same for the second inequality, and multiply it by d t B . Then we add-up the two inequalities, to obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B. Monte Carlo Experiments

In this section we study the behavior of our test in two Monte Carlo experiments. In the first one, we provide evidence for overrejection of the exponential discounting model by the deterministic test of Browning (1989) and correct coverage by our test. In the second experiment, we provide evidence for the power (consistency) of our test against some fixed alternatives. Finally, we conduct some robustness checks of our Markov Chain Monte Carlo (MCMC) integration.

## B.1. Overrejection of Exponential Discounting for Browning's Deterministic Test

The objective of the Monte Carlo simulation exercise is to test the performance of the methodological procedure developed in this paper against the deterministic benchmark. We are going to provide evidence that a data set generated by a random exponential discounter, when contaminated with measurement error, will be erroneously rejected by deterministic methodologies at the individual level for a sizable fraction of the sample (Browning, 1989, Blow et al., 2017). However, our test will not reject it.

We choose our simulation configuration setup to match those of the household characteristics in our application. The Monte Carlo exercise will deal with a moderate size data set of n = 2000 individuals to show that it works in a data set of the roughly the same size as in our application. The time period is T = { 0 , 1 , 2 , 3 } , and we consider L = 17 goods. We use the same discounted prices { ρ i,t } n i =1 as the ones given in Adams et al. (2014). 61 These are the prices faced by the single-individual/couples households in our application. We consider consumers with the constant

61 We use the observed price matrix and sample from it uniformly with repetition at each Monte Carlo experiment.

elasticity of substitution (CES) instantaneous utility

<!-- formula-not-decoded -->

where σ ∼ U [1 / 15 , 100] is heterogeneous across individuals.

Following Browning (1989), the true consumption rule for each consumer and each realized d is given by

<!-- formula-not-decoded -->

for all l = 1 , · · · , L and t ∈ T . For the discount factor we considered two different data generating processes (DGPs). For the first DGP d ∼ U [0 . 8 , 1] (DGP1) and for the second one d = 1 a . s . (DGP2).

We perturb the generated consumption with /epsilon1 t,l ∼ i . i . d . U [0 . 97 , 1 . 03] , which implies that E [ /epsilon1 t,l ] = 1 . That is, observed consumption is equal to true consumption times the multiplicative perturbation c t,l = c ∗ t,l /epsilon1 t,l . We define measurement error in consumption as w c t,l = c t,l -c ∗ t,l and fix w p t = 0 a . s . . Note that the implied random measurement error w c t,l is mean-zero by construction and satisfies Assumption 1.1. The random vector perturbations /epsilon1 t,l captures incorrect consumption reporting or recording, and can be as high as 1 . 03 times the true consumption. This means that relative measurement error is around 3 percent. This procedure produces a data set ( ρ t,i , c t,i ) i = n i =1 ,t ∈T .

We replicate the experiment m = 1000 times for both DGPs. The results are presented in Table 2. For the deterministic test in Browning (1989) we use a grid search over d on [0 . 1 , 1] with a grid step 0 . 05 . Searching over a smaller set (e.g., [0 . 8 , 1] ) will only weakly increase the rejection rate of the deterministic test. For DGP1 the deterministic test rejects the exponential discounting model in 53 percent of the cases on average across experiments. For DGP 2 the average rejection rate across experiments is 79 . 4 percent.

We use our methodology to test for s / ED -rationality for both DGPs assuming that the support of d is known. Assuming bigger support for d will only weakly decrease the rejection rate of our test. In other words, the design of our experiment favors the deterministic test. Nevertheless, our methodology cannot reject the correct null hypothesis that all households are consistent with s / ED -rationality at the 5 percent significance level. The rejection rate for each DGP1 and DGP2 is 1 . 2 percent. As expected, both rejection rates are less than 5 percent.

Table 2 - Rejection Rates: ED-rationalizability. Number of replications m = 1000.

|      | rejection rate ( % ) test   | rejection rate ( % ) test   |
|------|-----------------------------|-----------------------------|
|      | Deterministic               | Our methodology             |
| DGP1 | 53                          | 1 . 2                       |
| DGP2 | 79 . 4                      | 1 . 2                       |

## B.2. Power Analysis

We choose our simulation configuration setup to match Section B.1. However, the consumer units are assumed to be couples whose behavior is described by the collective model with exponential discounting described in Adams et al. (2014). The individuals in the household are indexed by A and B . The random discount factors are d A and d B . Individuals face different prices for good l at time period t given their bargaining power µ t,l . We observe the sum of these two prices ρ t,l . That is, ρ t,l,A = µ t,l ρ t,l , and ρ t,l,B = (1 -µ t,l ) ρ t,l . Note that the bargaining power is good and time specific. The random price vectors ρ t were drawn from the data set in Adams et al. (2014) as described in Section B.1.

Similar to the experimental design in Section B.1, the consumption rule for each individual and realized d j , j ∈ { A, B } is given by

<!-- formula-not-decoded -->

where σ l,j is a realization of σ l,j ∼ i . i . d .U [1 / 15 , 100] , j ∈ { A, B } . Then the household consumption data is the sum of individual consumption: c ∗ t = c ∗ A + c ∗ B a . s . . The generating process for measurement error coincides with the one presented in Section B.1. As a result we generate ( ρ t , c t ) t ∈T and test whether this data is consistent with s / ED -rationality.

d j , j ∈ { A, B } , and µ t,l . DGP3 . d A ∼ U [0 . 1 , 1] , d B ∼ U [0 . 99 , 1] , and µ t,l = 1 / 2 a . s . . Under this DGP household members face the same prices but may have different discount factors.

. d A ∼ U [0 . 1 , 1] , d B ∼ U [0 . 99 , 1] , and µ t,l ∼ i . i . d . U [1 / 3 , 2 / 3] . Under this DGP household members face different prices and may have different discount factors.

We consider two different DGPs for the distribution of DGP 4

We conducted the experiments with each DGP m = 1000 times for two sample sizes, n = 2000 and n = 3000 . The supports of the discount factors were assumed unknown and contained inside [0 . 1 , 1] interval. The results are presented in Table 3.

Table 3 - Rejection Rates: Collective Model. Number of replications m = 1000.

|      | prices    | discount factors   |   rejection n = 2000 | rate ( % ) n = 3000   |
|------|-----------|--------------------|----------------------|-----------------------|
| DGP3 | same      | different          |                   32 | 69 . 1                |
| DGP4 | different | different          |                   72 | 96 . 9                |

For DGP3 with equal bargaining power (same prices) and heterogeneous discount factors, the rejection rate is 32 percent for the sample size of n = 2000 and increases to 69 . 1 percent for n = 3000 . For DGP4 with asymmetric bargaining power (different prices) and heterogeneous discount factors the rejection rate is even bigger and is equal to 72 and 96 . 9 percent for n = 2000 and n = 3000 , respectively.

We highlight that DGP3 is compatible with hyperbolic discounting. It is easy to see that consumption behavior of the collective model with symmetric bargaining (i.e., same prices) satisfies the Afriat inequalities for hyperbolic discounting in Blow et al. (2013).

## B.3. Robustness of MCMC integration.

Our testing procedure requires some user-specified parameters: the distribution η and the length of the MCMC chain. As mentioned in Section 5.2, the choice of η has no effect on the value of the test statistic both asymptotically and in finite samples. In other words the difference in values of the test statistics computed using two different η 's can only be driven by numerical precision of the MCMC integration step and the optimization algorithm used. Thus, we focus on the performance of procedure for different MCMC chain length.

The results in Sections B.1 and B.2 were obtained using the chain length equal to cl = 10000 . We decrease the chain length to cl = 5000 and for the sample size n = 2000 we additionally experiment with DGP2, DGP3, and DGP4. The remaining elements of the simulations remain the same as before.

Table 4 shows that halving the chain length from 1000 to 5000 changes very little the rejection rates of the three DGPs of interest. This is of course desirable as lack of robustness would suggest that the MCMC chain has not converged. This provides reassurance that our choice of chain length 10000 is appropriate.

Table 4 - Rejection Rates: ED and Collective Models. Sample size n = 2000.

|      | rejection rate ( % ) cl = 10000 cl = 5000   | rejection rate ( % ) cl = 10000 cl = 5000   |
|------|---------------------------------------------|---------------------------------------------|
| DGP2 | 1 . 2                                       | 2 . 5                                       |
| DGP3 | 32                                          | 34 . 9                                      |
| DGP4 | 72                                          | 71 . 8                                      |

## C. Computational Aspects

In this appendix we discuss the computational aspects of our procedure. In Appendix C.1 we provide a general pseudo-algorithm to implement our procedure. Appendix C.2 describes the MCMC procedure used for latent variable integration. Appendix C.3 provides a description of the 'hit-and-run' algorithm we used in the construction of the MCMC chain. We provide the specification for η and the optimization routines used in our applications and simulations in

Appendix C.4.

## C.1. Pseudo-Algorithm

This pseudo-algorithm is based on Schennach's algorithm provided in GAUSS as a supplement to Schennach (2014). The actual implementation of the algorithm has been vectorized and parallelized.

- 1: Step 0 (Setting parameters)
- Fix T = { 0 , · · · , T } , consumer experiments, and L = { 1 , · · · , L } , set of goods.
- Fix g I and g M .
- Fix Λ , the support of ( λ t ) t ∈T , and ∆ , the support of ( δ t ) t ∈T .
- Fix η ∈ P E | X (See Appendix C.4 for details)
- Fix x = ( x i ) i =1 ,...,n , where x i = ( ρ i,t , c i,t ) t ∈T is i -th observation and n is the sample size.
- 2: end Step 0 .
- 3: Step 1 (Integration: Evaluation of the objective function at a given γ ∈ R |T | )
- Set i = 1 .
- 4: While i ≤ n
- Define the measure

<!-- formula-not-decoded -->

- Integrate latent variables using ˜ η ( ·| x i ) to obtain ˜ h M ( x i , γ ) (See Appendix C.3 for implementation details).
- Set i = i +1 .
- 5: end While.
- Compute

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Compute ObjFun( γ ) = n ˆ ˜ h M ( γ ) ′ ˆ ˜ Ω( γ ) -ˆ ˜ h M ( γ ) .
- 6: end Step 1 .
- 7: Step 3 (Optimization Step)
- Compute TS n = min γ ObjFun( γ ) .
- 8: end Step 3 .

## C.2. Latent Variable Integration

Evaluation of the objective function requires integrating latent variables. We use MCMC methods. For completeness we provide the algorithm for MCMC integration to get ˜ h M ( x i , γ ) .

## 1: Inputs

- Fix cl - total MCMC chain length; nburn - number of 'burned' chain elements;
- Fix η , γ , x i , and the first element of the chain e -nburn that satisfies the constraints.
- Set r = -nburn + 1 and ˜ h ( x i , γ ) M ( γ ) = 0 .
- 2: While r ≤ nsims
- Draw e jump = (( v t ) t ∈T , ( λ t ) t ∈T , ( δ t ) t ∈T , w c , w ρ ) proportional to ˜ η ( ·| x i ) = η ( ·| x i ) 1 ( g I ( x i , · ) = 0 ) .
- Draw α from U [0 , 1] .
- Set e r equal to e jump if [ g M ( x i , e jump ) -g M ( x i , e r -1 )] ′ γ &gt; log( α ) and to e r otherwise.
- if r &gt; 0
- Compute ˜ h M ( x i , γ ) = ˜ h M ( x i , γ ) + g M ( x i , e r )) / cl
- end if
- Set r = r +1

## 5: end While.

To compute the chain, one always can use 'rejection sampling': at every step check whether a candidate element of the chain satisfies the inequalities (support constraints). Since our constraints have a simple form, we propose to use a version of the 'hit-and-run' algorithm that we describe below.

## C.3. 'Hit-and-run' Algorithm

Since we use the algorithm presented below in our application with survey data and for concreteness we focus on s / ED -rationalizability with consumption measurement error. Note that instead of working with measurement errors in consumption, we can equivalently work with true unobserved consumption c ∗ t . Thus, the latent variables, e = ( d, ( c ∗′ t , v t ) t ∈T ) , have to satisfy the following set of constraints:

<!-- formula-not-decoded -->

for all s, t ∈ T .

The idea behind the 'hit-and-run' algorithm is (i) to pick some initial point e 0 that satisfies the support constraints; 62 (ii) to construct a candidate point by moving along a random direction within the constrained set on a randomly chosen distance; (iii) to use a user-specified Monte-Carlo acceptance rule to assign to e 1 either the initial point e 0 or the candidate point; (iv) to apply steps (ii) and (iii) to e 1 to construct e 2 ; (v) to repeat until the length of the chain reaches user chosen number.

Take some arbitrary e r that satisfies the constraints. Let ξ be a direction vector (not necessary unit vector). Thus, the candidate vector is

<!-- formula-not-decoded -->

where α ≥ 0 determines the scale of the perturbation αξ .

Sign Constraints. We start with sign constraints on consumption: c ∗ t,l ≥ 0 for all l and t . Let K c be a set of indexes that correspond to c ∗ t,l in e r . Hence, the constraints take the form

<!-- formula-not-decoded -->

Define K + = { k ∈ K c : ξ k &gt; 0 } , K -= { k ∈ K c : ξ k &lt; 0 } , and K 0 = { k ∈ K c : ξ k = 0 } . Then, the sign constraints are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the constraints that correspond to k ∈ K 0 are always satisfied since e r k ≥ 0 (i.e., satisfies the constraints). Thus, the sign constraints can be simplified to

<!-- formula-not-decoded -->

where min k ∈∅ = + ∞ , and max k ∈∅ = -∞ .

Afriat Constraints. Next, we consider the Afriat inequalities. Let e r ( v, t ) and e r ( c, t ) be the components of e r that correspond to v t and c t , respectively. Assume that d is fixed. Then, the Afriat inequalities are

<!-- formula-not-decoded -->

62 In our application and simulations we computed initial point by minimizing the norm of g M subject to the Afriat and sign constraints per observation.

<!-- formula-not-decoded -->

Thus, since d is fixed (the component of ξ that corresponds to d is zero), after plugging in e r +1 we get

<!-- formula-not-decoded -->

where ξ ( v, t ) and ξ ( c, t ) are the components of ξ that correspond to v t and c t , respectively. Hence,

<!-- formula-not-decoded -->

where P t = ( -1 , ρ ′ t /d t ) ′ , ξ ( t ) = ( ξ ( v, t ) , ξ ( c, t ) ′ ) ′ , and e r ( t ) = ( e r ( v, t ) , e r ( c, t ) ′ ) ′ . Similarly to box constraints, we can define T A + = { ( t, s ) : P ′ t ( ξ ( t ) -ξ ( s )) &gt; 0 } , T A -= { ( t, s ) : P ′ t ( ξ ( t ) -ξ ( s )) &lt; 0 } , and T A 0 = { ( t, s ) : P ′ t ( ξ ( t ) -ξ ( s )) = 0 } . Thus, the Afriat constraints are

<!-- formula-not-decoded -->

In other words we characterized possible perturbations of c t and v t given d that are allowed under Afriat constraints. Next we want to characterize the set for d given consumption and utility numbers. Note that by assumption d ∈ [ θ 0 , 1] and that

<!-- formula-not-decoded -->

Define T A d, + = { ( t, s ) : ( v t -v s ) &gt; 0 } , T A d, -= { ( t, s ) : ( v t -v s ) &lt; 0 } , and T A d, 0 = { ( t, s ) : ( v t -v s ) = 0 } . Hence, the Afriat ineqaulities are equivalent to

<!-- formula-not-decoded -->

Inequalities (5)-(7) give sharp restrictions on α that would guarantee that the next draw e r +1 satisfies the constraints. Below we provide algorithms how to generate (i) new consumption and utility numbers given prices and discount factor, and (ii) discount factor given prices, consumption, and utility numbers.

## 1: Generating new consumption vector and utility numbers

- Fix the discount factor and prices.
- Draw a random direction vector ξ from a uniform distribution on the [ |T | + |T | · L ] -dimensional unit sphere.
- Compute the interval A using (5) and (6).
- Draw α uniformly from A .
- Generate new consumption vectors and utility numbers using ξ and α .

## 2: Generating new discount factor

- Fix prices, consumption, and utility numbers.
- Uniformly draw d from the interval that satisfies (7).

Thus, we can propose the two approaches to sample from the cone characterized by the Afriat and the sign constraints. If one decides to keep the same d for generating the chain, then one can initially draw several independent draws of d , and for every realization of d generate its own chain. The second approach can be thought of as 'double-hit-and-run': first generate new consumption and utility numbers, and then generate new discount factor using these new consumption and utility numbers. In our application we use double-hit-and-run approach.

## C.4. User-specified ˜ η

In this section we specify a particular choice of η used in our applications and simulations.

When integrating measurement error, instead of drawing measurement error (e.g., w c t ), we draw unobserved true variable (e.g., c ∗ t ) and then constructed the measurement error by taking the difference between observed mismeasured and latent true variables (e.g., w c t = c t -c ∗ t ). Note that working with true variables allows us to easily generate measurement errors that imply correct signs for true variables (e.g., c ∗ t ≥ 0 ). In particular, in our applications and simulations we impose sign constraints directly in the sampling stage (Step 2 in Section C.2).

For our first application (survey data) to build ˜ η we used the 'hit-and-run' algorithm described in Appendix C.3 to produce draws of e . In particular, c ∗ = ( c ∗ t ) t ∈T is such that (i) it satisfies the Afriat-like inequalities and sign constraints, (ii) the user specified distribution over w c = ( w ∗ t ) t ∈T is

<!-- formula-not-decoded -->

where g M ( x, e ) = ( ρ ′ t w c t ) t ∈T . To achieve this, we use the standard Metropolis Hastings algorithm in each step of the 'hit-and-run' algorithm to get the draws from the desired distribution. Note that by construction this distribution has the correct support E | X .

The ˜ η distribution can be adapted to accommodate other moments such as those in our extensions and counterfactual analysis by using the appropriate moment conditions. If the moment conditions, which are not support constraints, include other random variables in e , the distribution ˜ η will have to be defined on them, and not only on w c like in our first application.

For our second application (experimental data) we use a different strategy since (i) the panel is long ( T = 50 ), (ii) the centering conditions do not depend on v t and δ t , and (iii) the static UMT has a simplified characterization in terms of Generalized Axiom of Revealed Preferences (GARP). 63 Hence, we can simplify our problem by considering a reduced latent random vector that consists only of true consumption or true prices. We then choose ˜ η to be a uniform distribution over

63 One can replace the Afriat inequalities by the GARP inequalities since GARP is equivalent to R-rationalizability in this case.

consumption or prices that satisfy GARP and that produce an expenditure level equal to 1 . 64 We can do this with the support constraints that we consider in this application. The key for a good computational performance of this step is to check for GARP consistency in an efficient way for each candidate draw of prices or consumption. For this purpose, we use a recursive algorithm to check GARP using an implementation of the deep-first search algorithm with recursive tabu search (see Boelaert, 2014).

In both applications it is trivial to verify that these choices of ˜ η satisfy the conditions stated in Definition 7.

## C.5. Optimization

We optimize the objective function specified in the pseudo-algorithm using Bobyqa procedure as implemented in the NLopt library following Powell (2009). As an initial guess for the optimization we use the outcome of applying BlackBox Differential Evolution Algorithm to minimize our objective function. Bobyqa performs derivative-free optimization using iteratively constructed quadratic approximations of the objective. We observe that in our simulations this combination of optimizers perform the best in terms of accuracy and speed among similar NLopt alternatives.

For the second application because the number of moments is larger we use as an initial guess for the optimization the outcome of two-step GMM estimator. Since the objective function of the two-step GMM estimator has a unique minimizer and is locally convex around it, we use Bobqya here as well. Bobqya works well in convex problems as documented in Rios &amp; Sahinidis (2013). Following Schennach (2014), we additionally verified our results using Neldermead. 65

Another alternative to find good initial values is taking advantage of a convex problem related to our problem. As shown in Schennach (2014), the moment condition in Theorem 3 is a first-order condition of the following convex optimization problem (Lemma A.1 in Schennach, 2014):

<!-- formula-not-decoded -->

Moreover, the norm ‖ E π 0 [ h ( x ; · ) ] ‖ has a unique global minimum, is convex in the neighborhood of the minimizer if this minimizer is finite, and has no other local minima. Hence, computationally the problem is convenient.

64 We impose nonnegativity constraints on consumption and positivity constraints on prices. The requirement to produce expenditure level equal to 1 makes the support of consumption and price bounded.

65 See Sasaki (2015) for an alternative optimization technique.

## D. Analytical Power Results. Robustness to Local Perturbations

In this appendix we provide examples of DGPs that will fail to pass our test (Sections D.1 and D.2). In Section D.3 we show robustness of UMTs that we consider to local perturbations in observed quantities or prices.

## D.1. s/ED-Rationalizability, Mean-budget Neutrality, Price and Consumption Measurement Error

In this section we construct the data set that can not be s / ED -rationalized by measurement error in consumption and time invariant measurement error in prices if the centering condition comes in the form

<!-- formula-not-decoded -->

where α ∈ (0 , 1] , represents individual specific weights. We consider the environment with 2 time periods and 2 goods. We assume that the price measurement error comes in the following form:

where

<!-- formula-not-decoded -->

is the matrix of time invariant multiplicative price measurement errors.

<!-- formula-not-decoded -->

The above centering condition covers variety of measurement error. For instance, if α = 1 a . s . , then we have the centering condition used in our application. The random weight α is allowed to be correlated with all observables and measurement error.

Take { c t } t =0 , 1 and { ρ t } t =0 , 1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote ρ 0 = (1 , 1) ′ , ρ 1 = 2(1 , 1) ′ , c 0 = (1 , 1) ′ , and c 1 = 2(1 , 1) ′ . By way of contradiction suppose that there exist d ∈ (0 , 1] , { c ∗ t , ρ ∗ t } t =0 , 1 , α ∈ (0 , 1] , nonnegative { v t } t =0 , 1 such that the Afriat inequalities and the centering conditions are satisfied:

<!-- formula-not-decoded -->

Note that since d &gt; 0 with probability 1 , the inequalities can be rewritten as

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, since d ∈ (0 , 1] , we can deduce that v 1 -v 0 ≤ 0a . s . If we multiply the first Afriat inequality by strictly positive α and take expectations from both sides, we get

<!-- formula-not-decoded -->

where the equalities come from from the centering conditions and the fact that ρ 1 = 2 ρ 0 , and the last inequality is implied by E [ α ] &gt; 0 . The above contradiction implies that the constructed data set will never pass our test.

There are at least two implications of the example constructed in this section. First, we can further restrict α by assuming that α = 1a . s . . Thus, the above example demonstrates that without price measurement error the centering condition E [ ρ ′ t w c t ] = 0 , t ∈ T has empirical content. Second, note that the trembling-hand centering condition ( E [ w c t ] = 0 , t ∈ T ) implies that in our example E [ ρ ′ t w c t ] = 0 , t ∈ T , since ρ t has a degenerate distribution. Hence, the trembling-hand centering condition has empirical content as well.

We conclude this section by noting that our example can be used to construct an example with time invariant consumption measurement error and time varying price measurement error because the mean budget neutrality condition and the Afriat inequalities are 'symmetric' in prices and consumption.

## D.2. GARP and Trembling-Hand Measurement Error in Consumption or Prices

In the experimental data we use individuals are forced to pick points on the budget lines. That is, ρ ∗ t c ∗ t = ρ t c t a . s . for all t ∈ T . In this section we construct an example for the GARP with trembling-hand error in consumption. Consider 2 goods and 2 time periods environment with deterministic prices.

<!-- formula-not-decoded -->

The observed consumption vectors are random and satisfy

<!-- formula-not-decoded -->

where 0 &lt; /epsilon1 &lt; 1 / 8 . Hence,

<!-- formula-not-decoded -->

Next, note that observed disposable income y t is a binary random variable:

<!-- formula-not-decoded -->

First time period. Since mismeasured consumption has to belong to the true budget line ( p ′ t c t = p t c ∗ t a . s . for all t ) and on average has to agree with the observed consumption, we can conclude that

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Hence, since c ∗ t ≥ 0 a . s . , E [ c ∗ 01 | y 0 ] ≤ 2 /epsilon1 a . s . . Thus, given that p ′ t c t = p t c ∗ t a . s . for all t we get that c ∗ 0 ∈ [0 , 2 /epsilon1 ] × [3 / 4 -/epsilon1, 1] with positive probability. Similarly, c ∗ 1 ∈ [3 / 4 -/epsilon1, 1] × [0 , 2 /epsilon1 ] with positive probability. Thus, since 3 / 2 -4 /epsilon1 &gt; 1 ( /epsilon1 &lt; 1 / 8 ) it means that c ∗ 0 is also available at t = 1 with positive probability. Similarly, c ∗ 1 is available when t = 0 with positive probability. The latter violates GARP with positive probability (GARP has to be satisfied with probability 1 ). Thus, there is no trembling-hand measurement error that keeps consumption on the same budget and is consistent with GARP.

We conclude this section by noting that GARP conditions are symmetric in terms of price and consumption vectors. Thus, after relabeling (swapping prices with consumption) the above DGP also will not pass our test if one assumes that there is only mean-zero measurement error in prices.

## D.3. Robustness to Local Perturbations

In this section we show that in many situations our approach is also robust to small measurement errors in observed quantities or prices. Suppose that we fix a model (i.e., the support restrictions on δ t and λ t , and the definition of ρ ∗ t ). Define the measure of inequality slackness

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Suppose that there exist { v t , δ t , λ t } t ∈T such that ξ t,s ≥ 0 . Next we perturb the true consumption and prices in order to see to what extent the RP inequalities are still valid. Note that the observed potentially mismeasured data { ρ t , c t } t ∈T satisfies the constraints with the same ξ t,s if

<!-- formula-not-decoded -->

for all t and s . Define

<!-- formula-not-decoded -->

Then by the triangular inequality, the following inequality provides a sufficient restriction on the maximal perturbations of consumption and prices that will not refute the correctly specified model.

<!-- formula-not-decoded -->

In other words, if the UMT leads to an 'interior solution' (i.e., ε ∗ t,s &gt; 0 for all t = s ), then small measurement errors without any centering restrictions will not affect the conclusions based on our testing procedure.

## E. Extensions of s/ED-Rationalizability

In this appendix we show that our methodology can cover two important extensions of ED -rationalizability discussed in the main text: (i) ED -rationalizability with income uncertainty (Appendix E.1); (ii) the collective model of Adams et al. (2014) (Appendix E.2).

/negationslash

## E.1. Income Uncertainty

In this section we consider a model of dynamic utility maximization with exponential discounting and income uncertainty. We start with the analysis of the deterministic model and then extend it to stochastic environments.

Definition 9 (Dynamic UMT with income uncertainty, ED -IU-rationalizability) . A deterministic array ( p t , r t , c t ) t ∈T is ED -rationalizable in the presence of income uncertainty ( ED -IU rationalizable) if: (i) There exists a concave, locally nonsatiated, and continuous function u . (ii) There exists a random income stream y = ( y t ) t ∈T . (iii) There exists an array of consumption and saving (policy) functions ( c t ( · )) t ∈T and ( s t ( · )) t ∈T such that c t : R |T | + → R L + \ { 0 } and s t : R |T | + → R + for all t ∈ T . In addition, we restrict these functions to depend only on the income history. That is, for all t , c t ( y ′ ) = c t ( y ) and s t ( y ′ ) = s t ( y ) for all y and y ′ such that y ′ τ = y τ for all τ ≤ t . (iv) The consumption and saving policy functions maximize the expected flow of instantaneous utilities given the budget constraints and history of incomes captured by information I t :

subject to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all τ = t, t +1 , . . . , T . (v) The consumption stream ( c t ) t ∈T at every time period t is equal to the consumption policy function evaluated at a realization y = ( y t ) t ∈T of the random income stream (i.e., c t = c t ( y ) ). (vi) There is initial level of savings s 0 .

ED -IU-rationalizability extends ED -rationalizability in an important direction: in accommodates possible uncertainty in the future income. Since the future income is unobserved, instead of a fixed vector of future consumption and savings, the agent has to come up with the whole consumption and saving functions in order to be ready for all possible realizations of the income stream.

In the case of income uncertainty we can still use the first-order conditions approach with an important modification. Instead of considering a support constraint on the space of marginal utility of wealth, we restrict its law of motion.

Lemma 3 (FOC for ED -IU-rationalizability) . A deterministic array ( p t , r t , c t ) t ∈T is ED -IU rationalizable if and only if there exists a concave, locally nonsatiated, and continuous function u , a discount factor d ∈ (0 , 1] , and a positive random vector ( λ t ) t ∈T such that:

- (i) E [ λ t +1 | I t ] = λ t a . s . , where I t the information ( σ -algebra) generated by ( λ τ ) τ ≤ t .

/negationslash

- (ii) d t ∇ u ( c t ) ≤ λ t ρ t a . s . . If c t,j = 0 , then d t ∇ u ( c t ) j = λ t ρ t,j a . s . , where c t,j , ∇ u ( c t ) j , and ρ t,j are the j -th components of c t , ∇ u ( c t ) , and ρ t , respectively, and ρ t = p t / ∏ t τ =1 (1 + r τ ) .

Proof. At every time period t the agent is maximizing the expected flow of instantaneous utilities given the budget constraints and history of incomes captured by I t :

subject to for all τ = t, t +1 , . . . , T .

The Lagrangian function of the above problem takes the form

<!-- formula-not-decoded -->

where { λ τ ( · ) } τ = t,...,T are lagrange multipliers. The denominator δ t ∏ τ j =1 (1+ r j ) is needed for scaling of λ τ ( · ) . If the instantenious utility function is concave, then, since the constraints are convex, the first-order condition will provide necessary and sufficient conditions for c t and s t to be optimal. The first-order condition with respect to c τ is

<!-- formula-not-decoded -->

for all t ∈ T , τ = t, . . . , T , and functions v c,τ , where ρ τ = p τ / ∏ τ j =1 (1+ r j ) . Note that, since for any j = 1 , . . . , L the first order condition with respect to c τ is satisfied with v c ( · ) = ( 1 ( i = j )) i =1 ,...,L , we have that the first order condition with respect to c τ is satisfied if and only if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all t ∈ T .

Next, consider the first order condition with respect to s τ :

<!-- formula-not-decoded -->

for all t ∈ T , τ = t, . . . , T , and functions v s,τ . Because of the law of iterated expectations the later is equivalent to for all t ∈ T , since v s,τ ( · ) only depends on the history up to moment τ .

<!-- formula-not-decoded -->

/squaresolid

The first corollary of the lemma above is that without imposing any restriction on income shocks, at the population level, it is impossible to discern whether an array ( p t , r t , c t ) t ∈T is ED -IU-rationalizable or R -rationalizable. The reason is that the only implication at the individual level of ED -IU-rationalizability is that the marginal utility of income is positive. 66 However, if we

66 This observation seems to have been noticed first by Adams et al. (2014).

assume that the latent income shocks are i.i.d., and there is no aggregate shocks, in addition to assuming that preferences are i.i.d. and stable in the time window of interest, we can still have testable implications of the model. This statistical version of the ED -IU model is defined next.

Definition 10 ( s / ED -IU-rationalizability) . Arandom array ( p ∗ t , r ∗ t , c ∗ t ) t ∈T is s / ED -IU-rationalizable if there exists a tuple ( u , ( λ t , d ) t ∈T ) such that

- (i) u is a random, concave, locally nonsatiated, and continuous utility function;
- (ii) d is a random variable supported on (0 , 1] interpreted as the time discount factor;
- (iii) ( λt ) t ∈T is a positive random vector, interpreted as the marginal utility of income, such that for t = 0 , · · · , |T | 1 :

<!-- formula-not-decoded -->

- (iv) d t ∇ u ( c ∗ t ) ≤ λtρ ∗ t a . s . for all t ∈ T ;

where ρ ∗ t = p ∗ t / ∏ t τ =1 (1 + r τ ) , t ∈ T ;

- (v) For every j = 1 , . . . , L and t ∈ T , it must be that P ( c ∗ t,j = 0 , d t ∇ u ( c ∗ t ) j &lt; λtρ ∗ t,j ) = 0 , where c ∗ t,j , ρ ∗ t,j , and ∇ u ( c ∗ t ) j denote the j -th components of c ∗ t , ρ ∗ t , and ∇ u ( c ∗ t ) , respectively.

/negationslash

In words, at the beginning of the time-window of interest, a consumer draws a utility function, and a discount factor that are going to remain fixed in time. Every time, given the realized prices, utility, and discount factor the consumer draws a new marginal utility of income, and chooses consumption according to her first-order conditions. The marginal utility of income is a martingale with respect to the known information, which includes the realizations of utility, discount factor, and discounted prices. With this definition at hand we can write the following lemma.

Lemma 4. For a given random array ( ρ ∗ t , c ∗ t ) t ∈T , the following are equivalent:

- (i) The random array ( ρ ∗ t , c ∗ t ) t ∈T is s / ED -IU -rationalizable.
- (ii) There exist positive random vectors ( v t ) t ∈T and ( λt ) t ∈T , and d supported on or inside (0 , 1] such that

<!-- formula-not-decoded -->

and such that for t = 0 , · · · , |T | 1 :

<!-- formula-not-decoded -->

The proof of Lemma 4 is omitted as it follows trivially from our previous results.

Econometric Framework.The additional restrictions implied by the income uncertainty in classical ED -rationalizability can be captured as a set of conditional moment conditions that restrict the latent distribution of the marginal utility of income. Note that our framework so far has only dealt with unconditional moments and support constraints. Fortunately, the ELVIS framework can deal with conditional moments too (Theorem 4.1 in Schennach, 2014). The main intuition behind this extension is that a finite set of conditional moments can be written as a (possibly infinite) collection of unconditional moments.

For simplicity of exposition and for practical purposes, instead of the martingale condition from Lemma 4, we will use its simplest implication:

<!-- formula-not-decoded -->

for all t ∈ T \ { T } . Moreover, in order to be able to take expectations of marginal utility of income in the cross-section of individuals, we need to impose a normalization condition such that the marginal utility of income of all individuals is in the same units. A natural normalization (without loss of generality) in the form of a support constraint is

<!-- formula-not-decoded -->

Recall, that in the benchmark case of perfect-foresight ( s / ED -rationalizability), the marginal utility of income is normalized to 1 for every time period (i.e., λ t = 1 a . s . ). In the case of the static UMT ( R -rationalizability) λ t is only restricted to be positive. The s / ED -IU-rationalizability provides a framework that is less restrictive than s / ED -rationalizability but is more restrictive than R -rationalizability.

Empirical Results.In our first application we rejected the null hypothesis of s / ED -rationalizability with perfect-foresight for the case of couples' households (at the 5 percent significance level). However, we fail to reject the implication of s / ED -IU-rationalizability captured by the above moment conditions for couples' households at the 5 percent significance level with a discount factor set at d = 1 a . s . . We find that TS n = 9 . 047 (p-value = 0 . 249 ) is below the 95 percent quantile of the χ 2 7 ( 14 . 07 ).

We have not tested all necessary and sufficient conditions for s / ED -IU-rationalizability. But the evidence we provide suggests a possible explanation of the rejection of the perfect-foresight s / ED -rationalizability. In short, it may be that couples' households face more income uncertainty than singles. Hence, not taking income uncertainty into account could be a reason why we reject the dynamic UMT in the couples' households case. Indeed, Browning et al. (2010) points out that risk sharing may be a benefit from marriage. Further exploration of this explanation is beyond the scope of this paper.

## E.2. Collective Exponential Discounting Model

The important contribution of Adams et al. (2014) studies a dynamic collective consumer problem to model the behavior of couple's households. The collective model considers a case in which the household maximizes a utilitarian sum of individual utilities of each member of the couple over a vector of consumption of private and (household) public goods, given the individuals' relative power within the household (Pareto weights). Each individual member of the household is an exponential discounter but the observed consumption is a result of the collective decision making process, and may not be time-consistent. We formulate a test for the collective model using our methodology. We fail to reject the null hypothesis of consistency of the data set with the dynamic collective model assuming that the random discount factor is supported on [0 . 975 , 1] (this support is the one used in Adams et al. (2014)).

Consider a household that consists of two individuals labeled by A and B . Partition the vector of goods into publicly consumed goods indexed by H and privately consumed goods indexed by I . That is, c t = ( c ′ t,I , c ′ t,H ) ′ and p t = ( p ′ t,I , p ′ t,H ) ′ . Let c t,A and c t,B be the consumption of the privately consumed goods of individuals A and B , respectively ( c t,I = c t,A + c t,B ). Then the collective household problem with exponential discounting corresponds to the maximization of

<!-- formula-not-decoded -->

subject to this linear intratemporal budget constraint:

<!-- formula-not-decoded -->

where ω A , ω B &gt; 0 are Pareto weights that remain constant across time and represent the bargaining power of each household member. Individual utility functions, u A and u B , are assumed to be continuous, locally nonsatiated and concave. The individual discount factors are similarly denoted by d A and d B . The rest of the elements are the same as in our main model.

The quantities c t,A , c t,B are assumed to be unobservable to the econometrician. We observe only c t . Adams et al. (2014) propose one solution to the collective household problem above. They assume full efficiency in the sense that there are personalized Lindahl prices for the publicly consumed goods p t,H that perfectly decentralize the above problem. The Lindahl prices are p t,A ∈ R L H ++ for household member A and the analogous p t,B such that p t,A + p t,B = p t,H . Adams et al. (2014) established the result which is the analog of Theorem 1. Similar to the case of the singleindividual household, define ρ t,h = p t,h / ∏ t j =1 (1 + r j ) for h ∈ { I, H, A, B } .

Theorem 6 (Adams et al., 2014) . An array ( ρ t , c t ) t ∈T can be generated by a collective household exponential discounting model with full efficiency if and only if there exist d A , d B ∈ (0 , 1] ; strictly positive vectors ( v t,A ) t ∈T , ( v t,B ) t ∈T ; individual private consumption quantities ( c t,A , c t,B ) t ∈T (with c t,A + c t,B = c t,I ); and personalized Lindahl prices ( p t,A , p t,B ) t ∈T (with p t,A + p t,B = p t,H ) such that

for all s, t ∈ T :

<!-- formula-not-decoded -->

With this result in hand, we can establish our finding in a very straightforward manner. We let ρ t and c ∗ t be the random vectors of deflated prices and true consumption. Finally, we define d A and d B as the random discount factors for household members A and B , respectively. Also, u A , u B and ω A , ω B denote the random utility functions and random Pareto weights for each household member. We keep here the assumption about the data-generating process that we maintained for the case of s / ED -rationalizability, namely, we assume that the preferences and Pareto weights remain stable for each household after being drawn from the joint distribution of ( u A , d A , ω A ) and ( u B , d B , ω B ) at the first time period. Now we can establish and prove a stochastic analogue to the result in Adams et al. (2014).

Theorem 7. If a random array ( ρ t , c ∗ t ) t ∈T is generated by a collective household with random exponential discounting under full efficiency, then there exist random variables d A , d B which are both supported on or inside [ θ 0 , 1] , and strictly positive random vectors ( v t,A ) t ∈T , ( v t,B ) t ∈T that satisfy

<!-- formula-not-decoded -->

Theorem 7 does not provide sufficient conditions for collective rationalizability. We can provide a stochastic analogue of Theorem 6, but our choice has several advantages: (i) one does not need to specify which goods are consumed privately or publicly; (ii) the inequality restrictions in Theorem 7 do not depend on the unobservable Lindahl prices and private consumption vectors, which simplifies implementation; and (iii) we can maintain Assumption 1 in a very natural form. We also assume that prices are measured precisely. Assuming that d A and d B are supported on or inside [0 . 975 , 1] we find that TS n = 0 . 018 (p-value &gt; 1 -10 -4 ), which is below the 95 percent quantile of the χ 2 4 ( 9 . 49 ). Thus, we fail to reject the null hypothesis that the couples' household data set is consistent with the collective exponential discounting model under the assumptions of full efficiency, common support for preferences, and the collective mean budget constraint. The test statistic value for the explored θ 0 = 0 . 975 for the collective model is below that of the exponential discounting model for the sample of couples' households.

## F. Empirical Application (I) Extended: Average Varian Support Set for Budget Shares

Here we compute bounds on counterfactual average budget shares. Since the null hypothesis of s / ED -rationalizability cannot be rejected for the case of single-individual households in our

Figure 1 - Average Support Set for Budget Shares: Petrol

![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA9QAAAC6CAIAAADAoJJkAAB31ElEQVR4nO3dd1gUV9Qw8JntBVh6701AQBHErti70RiNvUZjjyUajYkmaiyxR6PG2GJMDHYSu2IvICoWlLb03lm2l5n5nnfv8847HxpjjLCU8/vDx73bZmeXO2funHsuTlEUBgAAAAAAAKh7rHp4DwAAAAAAAAAE3/8CRVEGgwEuFCAURZEkaeqtaCgMBgNBEKbeioaCJEn4M0FIkoROg0YamXorGtDRBPYGAkcTJoIgDAaDqbcC1DkIvt+WXq+XyWR6vd7UG9IgGAwGjUZj6q1oKJRKpVqtNvVWNAgURanVajiUIjqdrqamBk7MEL1er9PpTL0VDQJJknK5HLpQhCRJtVoN56iIRqORy+WwN5q8ugq+q6qqqqur/+5ejUZTVlbGPL1DIwEI81il0WhKS0sbyIkg/D0wwd6gUUam3oqGAnYFDX4YTLA3mGBvMMGuoMEPo5ngvPdX1Gq1p0+fjo+Px3G8Y8eOH3zwAZfLZT4gKSkpOjpaLpc7ODiMGTPGw8MDw7AzZ87Ex8ez2WyCINzc3CZMmGBubv7w4cOYmBi5XG5nZzd69Ghvb2/MdHj/y4TbAAAAAAAAGrX3P/J97dq1e/fuTZs2bcKECTdu3Lh16xbz3tLS0kOHDgUGBi5evFgsFv/yyy86o4SEBIlE0qZNm7CwMH9/fz6fL5VKDx06FBkZuXjxYj6ff+DAARNepFMqladOndqxY8eJEyeUSqWpNgMAAAAAADRq73nk22AwJCQkdO7cOSgoCMOwjh07JiQkREVFsdls9IDU1FSCIAYPHmxubj5s2LCNGzdmZ2fb2Njo9foPP/ywRYsW9EvFxcX5+fkNHjyYIIipU6dKpVLMRGQy2WeffXb48GF0Mejjjz/++eefzc3NTbU9AAAAAACgkXrPwbdKpaqurnZ2dkY3nZ2dnzx5otVqRSIRaikuLra2tkY3JRKJUCgsLS2lKKq6uvr06dM1NTWWlpYjRozw8vIqLCyUSCR79uzJzMz08PAYNWqUQCCo/4nSOI4fPXr0l19+oVuio6PDw8MXL17cnOdRoYoWzXkPvJqlB3uD/nuEXYGgrokgCBzHTb0tpgedBg3tBNgbCEEQqNOAP5P67zRYLBbs9iYy8k1RFJ0YzeVyCYJgVgjRaDR8Ph992SwWi8vlarXaiooKHMeDgoLc3NxiY2N37969YMECtVqdnJw8fPjwyMjIs2fP7tmzZ9GiRa/G3zqdru7yQHAcpyjq+vXrtdpjY2Pnzp2Lqjo02x8uRVFQu4BZQ00mk5l6QxoEiqL0en2z/bt49awMEtUQdOVQq9WaekMaBJIktVotdKEIRVE1NTWm3ooG1GnUz95gsVjm5uZ0YgJoxME3m81msVh0cRJ09sb8ank8Hl34FlU4wTCsQ4cObdq0QYG1h4fH119/nZyczGKxQkNDBw0ahGGYubn51q1b8/PzfX19X31HPp+P1Q0UQNAD+TRnZ2cOhyMSib755huKopYuXcrj8SiKwnGcz+eTJNnku1Q0UFFrKm2zpdVqcRyHybiIXq/ncDgQfKPBCL1ez+VyWSwo6vo/hwOKojic9z/Lv9GhKEqr1bLZbOhC6UgAdgWi0+lIkqy7qIaJZVQPbwRe9Z77QaFQKBKJqqqq0M3KykozMzPmcLW1tXViYqLBYGCz2Wq1WqPR2NrapqWlURQVEBCAonN0rLKysqJ/FgKBgM1mv7bgIMcIq0vjxo07ceJEfn4+/RFmz57N4XDYbLZGo6mqqrKwsECbqtFotm7d6uvrO3ToUKxJMxgMWq1WLBabekMaBL1ez2azYW8gCoVCJBJBn45hmFqtNhgMIpEIxpZQ90hRlFAoNPWGmB4aoOFyudBpoL2hUqlgVzAvKcPeaPLe8wGSx+O1bNny7t27JSUlRUVF9+/fDwkJ4XA4paWlmZmZJEkGBAQolcp79+7J5fKbN29yuVxfX1+pVLpr16709HSFQhEbG0tRVGBgYFhYWFJS0osXL2pqam7cuCESiVxcXDBTCA8P//333wcPHhwaGjpw4MATJ06Eh4ejcfFNmzbt27ePjjNkMtnFixcfPHhAP7eqqurs2bOZmZlY0wKFSMFr0Re1TL0hDQjsDVAL/CSYoNN4FeyNJu/9jxn36dMnMzNz7dq1FEW5urr27NkTw7D79+8/ffp00aJFrq6uAwYMOHbs2IULF9Rq9dixY8VicVRUVHp6+rZt28zMzFQq1ejRox0cHCwtLaVS6Y8//igSifR6Par8jZlIly5d2rVrl5mZ6enpyRzIrzWgZWtr+8svvzAvn6WkpIwdO3br1q10kfLq6urS0lJvb2+4/AoAAAAA0Nz8z4TC9/6iBoMhMzMTx3EvLy8UYur1eoIg6KmWpaWlxcXFbm5uVlZW6CkkSebl5cnlcnd3dwsLC/qlCgoKqqur3d3dTV7aT6/XKxQKMzOzf5WaptFoXr586eLi4uDggFqOHTv28ccfP3jwoG3btqilpqZGrVbb29s3ojRZtFI0XBpDqqur2Wy2yX+iDQGaXygUCiHRAqWdqFQqiUQCZ9qQdsJEEIRMJuPz+dCFor2hVqvFYnEjOgLWHYVCodPprKysYG80bXVySOBwOP7+/swWrhF9096I+QAWi4WWuqzFxQhrAFBZqH97riIQCNq0acNsiYyM3L9/f2BgIN1y4MCBw4cP3759m+6IUbEhyJoFAAAAAGhiYDymvnl6ek6ZMqVWS8+ePemxMZIkv/jiCxzH169fD/E3AAAAAEBTArGd6Q0dOnTjxo3M0kK4EX1Tq9V+/vnnZ86cMdEGAgAAAACA9wNGvhscFou1fv16ZtqJwWC4fv26m5sb/RiFQhEdHd2zZ09PT0/TbSkAAAAAAPh3IPhuiGplm4hEogsXLjCrrLx8+fKTTz6Jjo6mg++SkpK8vLyQkJD6Kc4PAAAAAADeAaSdNAI4jtvb2zOLwLRp0yYtLa1Pnz50y9mzZ3v16pWYmEi3lJaW5ufnEwRR79sLAAAAAABeD4LvRonD4fj5+VlaWtIt3bt3P3LkSIsWLeiWrVu3Tpw4sby8nG4xGAy1YvFbt26NHTt24MCB3333XWVlZX1tPgAAAABAMwVpJ02EtxGzpV27djY2NhKJBN0kCGLKlClWVlbbtm1DszkvXLgwbtw4FHOfP3/+/v37J06cYCa3AAAAAACA9wuC7yZr6NChzJskSXp7e9N1VAwGw9atW5mj3eeMhg8fboqNBQAAAABoFiD4bi64XO4333xDLxKk1+tTUlJqPWb37t0qlapjx44+Pj6m2EYAAAAAgCYOcr6bF7p8OI/HCw8PZ97F5/PDwsLmzZv39OlTujEvL+/KlStqtbretxQAAAAAoAmC4LuZYrPZS5cuZZYJHzdu3Hfffffy5cv+/fvTjbGxsQMGDCgsLKRbCgsLU1NT6RF0AAAAAADw9iDtpPlq167duXPnTp48WVJS0qFDh6FDh/J4PCcnJ+Zjevbs+eeff3p4eNAt27dvv3bt2vXr19lsNmrR6/UoraXePwEAAAAAQCPT6INv0qge3ggV6SMIwmAwYE1FkBFJkiwWi6KoVz+amxHzU0dFRXl6euI4ThjpdLqRI0d6enpu2bIFvQiqhIjmdNKvg+N4Ex4sp4ya0g/jvyBJkiCIJvx1vz3UNUGtfYQkSfgzYf4wSJKEvUEfxOHPpP47DRzHWSwWnYwK6lOjD74JgtBqtfX2J6HT6ZpYH0H/4b1ltNSjRw8Wi6XRaAiCQP+2bt3azs5Oq9WiUipKpXLKlCkjjNDhFsdxDofThLtX9NvQaDSm3pAG9CcJHTp9/qnVamutWduc9wacldGn66gLNfW2mB7sCiY0clE/e4PFYgkEAuirTaLRB99cLheNs9Y1vV6v0+mEQiHkV2AYJhAIdDqdSCTCcXz16tX0MRUF31VVVVwuVygUosaCgoL9+/ePGzeOrkT+byP+Bs5gMLDZbLFYbOoNMT2KolQqlVAohHATnY+hvVE/fVQDp9VqKYqClQRQgKXX67lcrkgkMvW2mB6KvNHRxNTbYnpKpZI+ttb1e8EON6GmcEio5x8Q/F5fuyvo/9vZ2V2/fp3ZmJ6evnXr1sjISLqCYXJycnp6evfu3c3NzbGmAn4YTLA3aHRxfQA/jFo7AfYGvRPgz4QJ9kaTB6NT4P3jGtE3u3TpkpOT06NHD7rl0qVL8+fPz8nJoVukUunLly8hAxIAAAAATVtTGPkGDRyHw6FXuUeGDx/evn17OgsFw7DVq1fn5ubGxMRYWFigFrVazWazeTxevW8vAAAAAEBdgeD7bbH+l6k3pClARVSYLZMnTy4tLaWzUNRqdd++fVu0aPHzzz8zpyjB/gcAAABAowbB99tSq9UVFRUcDocemgXvUVRUVK2WkSNHMjPC8/LyFi5cOGTIkAkTJrzNC1ZXV2MYZmlp+b63FAAAAADg3cE44j+jKOrkyZNdunRp3bp1x44djx49auotavqEQuGcOXMmTpxIt+j1+lqlEpKTk5ctW5aWllbrucXFxTNnzvT09HR3d58yZUp+fn49bjgAAAAAwJvAyPc/u3PnzpQpU2pqajAMe/HixbRp0xwdHbt3727q7WpefHx8Tp48yawUXlRUtH///oEDB/r7+6OW58+f37t378aNG3/88QdqOXjwYFVV1YkTJ+j1OAEAAAAATAiC738WHR2NIm9EqVQeOHAAgm+TYMbQ3bp1y87OFgqFdMvt27eXLl0ql8uZT/nrr78ePXoUGRlZv1sKAAAAAPAaEHz/M6VSWauFDu/u3LmD43jLli3NzMxgHY16xmazay1RMXz4cFtb2zFjxjAHyAmCoM+dKIqqrq4WCoWw0gcAAAAATAJyvv9Znz59ahXZ6NOnD4rktm/f3rlzZ1dX16ioqPnz5x8+fPjWrVu5ublNY+HGRsfBwWHEiBG1BrmDgoI6duyI/q/Vajt16rRw4UL6XoqiXj25AgAAAACoIxB8/7PBgwfPnz+fz+fjOM7n82fNmjV58mR014oVK37//ff58+cLhcIjR45MnDhx4MCBPXr06N+/f1VVFYrttFotcyAW1Ckcx7du3RoSEsJms1ksVkBAwK5du+gBcg6HM3v27EGDBtGPr6ysjIqKOnbsmOk2GQAAAADNCGRK/DMzM7PNmzePGTPm4cOHbdq0adu2LWrHcTzECOU2aLVaqVT68OHDuLi46upqPp+PYdijR4/Gjx/v5uY2cODAuXPnslgskiShWHWdateu3Z07d27evEmSZLdu3ZjVBlHwzXywRqNxdna2srKiW548ebJz584lS5bQ8zgBAAAAAN4XHBIk3hJJkgqFwszM7C1DZ4qicBzPycnZtm3bnTt3fHx8fvvtNzabfe3atXXr1qEgPiIiwtHREY2pY42KXq/X6XRisRhrci5fvrx48eLdu3fTySo3b95MTk4eNWrU31UNr66uZrPZzKrkzRZK4xEKhVBeBi0OoFKpJBIJTAhBJ7oURTFnSDdbBEHIZDI+n98ku9B/iyAItVotFosb3XGwLigUCp1OZ2VlBXujaauTQ0JSUtKNGzfYbHZUVFRgYGCte6urqy9fvpyXlxcYGBgVFYVSAp49e5aeno7jOEmS1tbWHTt2FAgEycnJt2/flsvlQUFBXbp0MTMzw0zHYDDodDqDwfCWC56jvxwPD4+tW7fq9Xq6UodarZbL5b///vu2bdt0Op2Hh0dkZGSbNm2CgoLatm3r5ORU9x8FvEmvXr0SEhKY0dLdu3fPnDnTv39/OvhOSEggCCIiIoJ+GMSaAAAAADDNyPfz58/37NnTrl07nU735MmT2bNnM+NvlUq1e/dumUwWHh5+8+bNkJCQSZMmkSS5Zs2anJwce3t7g8Hg5uY2bdo0qVS6c+fO8PBwZ2fn2NhYf3//6dOnm3AASafT1dTUWFhYvGXw/QZarTY7Ozs3NzctLe3x48fx8fGpqakGg+H48eMfffQRSZLnz5/HcTw8PNzOzq5hRnVNeOT7VWVlZZWVlT4+PvTPr1+/fhqN5sqVK1wuFyWOl5aWurq6mvb8sCGAkW8mGPlmgpFvGox8M8HId2Mf+dbpdH/88YednV3//v1NvS2Nxns+JFAUdeXKlZCQELQG+M6dO2NjYwMCAuifUWpqalZW1vLly52cnLy9vffu3dunTx9zc3OlUjlv3rzWrVvTCRsvXrxo06bNp59+imGYq6vr3r17y8vLHR0dscaPz+e3MOrduzf64VZVVT19+rRly5Yov2X37t3nz5+3tLS8ePFiu3btCIKIj4+3srLy8vKCGnn1z86I2bJ58+bq6moUeaPge8iQIZ07dz5w4ABqMRgMNTU1EokEYlAAAABNWH5+/po1a+bOnWvqDWnGwbdKpSoqKhowYAC66e/vf/36da1WS4eMubm51tbWKJRxdXXl8/n5+fkODg46nS4jIyMpKcnJyalLly48Hq9fv350yF5TU8PhcOhYp4nh8XgODg6ofCFKYPj++++nT59eUlLi6emJYVhVVdXw4cPZbLarq6uvr2+EUXBwsEgk4vF46NpFIzpLbgLQaRJNJBItW7bM3d2dbklLS5s5c+ann346ZswY1EIZ4Ub1vr0AAADA+0eSZFZWll6vLykpKSwsJAiCxWK5uLigeymKysnJMTc3x3FcrVabmZldv35drVZ36dLF1dWVfpGqqqo7d+5UVlaGhYWFhoZizcB7Dr61Wq3BYKAru4lEIpSfQAffcrlcLBajOYtcLpfP56N4PT8/Py0tzdbW9syZM8+fP58xYwZdgCIjI+PEiRNdunSxsbF57Tuq1Wqs7pEkiRbcUalUdf1enp6evr6+BEHodLrq6mqDwbB+/fpHjx49f/48ISHh7NmzMplMKBQGBQUNHz585syZOI5rtVoOh2NmZqbT6fR6fV1HeBRFkSSJctmBQCAYMWIEWsEHnT6hbH5zc3O5XE4QBI/Hu3379p49e5YvX+7v70+SJEVRXC6Xw+Gga/FNKSInCKIefoGNAkEQFEXJ5XLYG3QXqtVqTb0hpof6T41GA10ovTcMBoOpN6QBdRoymawe3ovFYr19DYm/U1ZWtnz58vz8/B9//LGsrEwikVy8ePHKlSsODg5oOt9HH320efPmjIyMQ4cOOTg45OXl6Y12796NBmrv3r07a9YshUJhYWFRUFAwderUr776qslnZL3n4BsN7DHzyGvllL/6NRsMBj8/v6VLl4aGhvJ4vLS0tE2bNr148SI8PBzDsKdPn+7bty80NHTIkCGvfUeWEVYvSJLEcbwe3o4wot9LLBZ//PHHY8eO1ev16EQlLy8vKSkpPj4+Ly8PndicOHFi7969fn5+H3/8cc+ePVFHhi4XUBRlMBjQn3StN2Kz2Twej8Vioemkbz8BAD0SaiYi6DSJw+GgHUJRlK+v74EDB1CJdxaLxWaz9Xp9dnY2h8NBiSgcDufSpUt37txZsmSJSCRC+xN9FxqN5rXvguM416iBF49HxTQh3KSjCtgbCHQaTOiHAXsDoSgKdgXzHLV+9sZ7eRdra+uvvvpq9uzZI0aMmDVrVnp6+v79+2/evDly5EgMwy5evCgSidq2bZuamvrs2bMJEybs2rXLYDAsWLDgyy+/DA8Px3F84cKFPj4+GzZssLKyunTp0sKFC/39/enVVJqq9xx8CwQCHo9HLxmoVCoFAgGqeI1YWFgoFAoUkeh0Oo1GY25uzufzg4KC0ERGV1dXsViMRhBv37595MiRPn36fPjhh3939ELhCFZfEy5RpgdmIhwOx9sI3aQjbAzDJBKJUCiMj4+PjIxEVx52796dnZ3dunVrLy8vDw+P19ZRqampiY2NlcvlAQEBqGD5W2pWEy7/0d+VGmROsxswYECfPn2YPx6pVBobG7t69Wr0fVEUdffuXYPB0LVr17/rE5OTk5OSkkQiUceOHZm1yRsOmHD56oRLsVgMEy5hwuWrEy55PB50oTDh8rUTLlGeBtYYcLncli1bCoVCX19fb29va2vrFi1anDp1avjw4Wq1+vLly927d3dyctLr9U5OTl9//bWvry+GYV9++eXQoUMfPHhAUVR6evr8+fPZbHZ1dXVERESLFi1OnDgxfvz4pt1tvv/g29XV9cWLFz179iRJMikpyc3NjcfjkUYcDsfDw+PixYsFBQWenp6ZmZk6nc7Ly+v69etpaWlz5swRCATZ2dlqtdrV1TU5Ofm3334bOnRor1690CAfm81uLD/H+sH8aX5klJubSxfckEqlu3fvVqvVlpaWDg4OHh4ewcHBkZGR4eHhXl5ebDY7Pz9/7ty5MTExFEW5uLgsX7585syZpvs0TRyLxap12jZ9+vSRI0fSGVkkSa5YsYLFYl26dIluSUlJcXJyQnH24cOHV6xYkZOTg2FYnz59fvzxR9SLAQAAAKai1+vRBXYMwywtLfv27Xvw4MHs7OzKysqMjIxly5ahUyw3NzdnZ2f0FA8PD5FIlJ2djeO4XC7/+uuv6fGaiooKf39/jUbTtKuHvf+0k969e+/cuXPz5s0UReXl5c2bNw/H8QsXLiQlJc2ZM6dFixYBAQHbtm0LDg5OTEzs3Lmzg4NDq1atYmNjN2zY4Obm9ujRo44dO3p6eq5fvz4nJ+fPP/+MiYkhSdLCwmLRokVQBvvNmHP+1q9fv2LFitTU1Pj4+EePHqWmpp44cWLnzp1sNjslJcXNze37778/c+YMenBBQcHChQu7du1aayohqDsSI/omm80+ePCgSqWih701Gk1UVNTYsWO3bt2akZExd+7cmpoadNfly5e//fbbQ4cOwQAzAACAhmPIkCE7duy4fPlyaWmps7Nz+/btUTvzii5pzK7hcDh6vd7S0nLVqlWOjo4oM43P56Mr+ViT9v5H9Vu0aDF//vwHDx5gGDZ8+HAvLy8Mw4KDg52dnVGKyKRJk+7du5efnz9q1KjIyEh0DrRkyZL4+PiampqxY8e2bduWzWYPHTq0Z8+e6MugKIrH4zEjFfCPuFyuRCKJNELznPLy8rKysoqLi52dnXU63fnz55mP12g0y5YtmzhxooODg7e3t5WVVROuMNMweXh4MG9yudydO3eiVe5v3LhBR97I2bNnUeZ3Xl6et7c31KAEAABQ/1BKAn0pPigoKDw8/LffflMqlcOHD0cD2CwWKz8/v7S0FNVwS0tL02g0/v7+KpVKp9PZ2Nj06NEDPf3mzZtqtbrJjyvVSUqNjxGzxd0I/d/MzIwuq0dzcXH58MMPmS2tWrWqi21rtvh8vq8RukmSZK2TGTabrVQqZ8yYUV5ejmGYvb29t7c3+iq9vLzCw8NRUjgqzcHlchvshL8mg8vlojkr6FperXvNzMx4PN758+dHjBhx/vz5nj17ovaUlJS0tLSuXbu++hQAAADg/UKDdGlpaQUFBc7Oznw+/8MPP5wxY4ajoyO97A7KdP3mm28WLlyoVCpXrlwZGhratm1brVYbGBj4xRdfcLlcFxeX27dvL1q0aM6cOWjQsAlryvns4A14PN7kyZMfP35Mt3h7e+/evbukpKS8vDw/Pz8jIyMzM/Px48enTp1Sq9VfffUVCr73799/7NgxV1fXiRMnduvWDY2po8J5Tf5U1YR69uwZGhr67NkzumXWrFkcDicsLOzXX39lJgudPn16+/btV69epYPvY8eOZWZmzpw5kz7dqs8J9QAAAJowOzu78PDwgwcPFhcX//LLL1wut1u3bm5ubiEhIUFBQegxJEk6OTnl5OQMGzZMo9F4enpu3rzZwsICw7AdO3bMnTt39OjRlpaWJSUlH3744bx587CmDoLv5mvGjBkajebo0aMymSwgIGDlypX+RvQDUCVvnU6Xl5dHx22o6svNmzf79euHHvPJJ5/cvHkzICAADZO7u7vb29vb2NjY2dnZ2to27QnL9cbS0vK3335buXLls2fPxGLx8OHDFy5ciGGYmxHzkZ9++umgQYOYl56ePn368OHDadOm0S1ffvmlwWBYt24dnVZUUlIiEAggswsAAMC/IhaLf/rpp4cPHwoEAnTEt7S0tLe3HzRoEJ26TVGUpaXlL7/8UlhYiFIb6Lvatm1748aN58+fl5aW+vr6+vv7N4fSGhAYNV8cDufzzz+fMGGCTCbz8vJ6NUpmsVh8I/rkFQ24Tp06taCgACVy4Tjep08fPp8vlUrPnj1bUFBAUZRIJLKwsPDy8tq3b19QUJBerz927JitrW2LFi3c3NxggPzdBAcHHz9+PDMzUywWv2HmsbURs2Xx4sUqlYouTYgSh5hJdSqVasaMGW5ubj/88ANq0el0WVlZdnZ2tV4KAAAAqMXc3Lx79+70qhdnzpxRq9V0zgk9licUCun5l0wCgaBt27ZYcwLBd3Nnb/SvnsLn8728vNBaMDiOjzcyGCkUiuzs7IyMjLS0NKVSiS4qqVSqL7/8Mjc3Nygo6PLlyy4uLiUlJT/88IOrET1G3jALVzcoLBbrHcoLWhrRN3Ec/+6772qtatGuXTtmnJ2fn9+3b9+PPvpo06ZNqKWysvL27dshISF0mfnXag4jFgAAAF4rMTFx0qRJBQUFa9asQXMrETMzM1tbWzhA0P6/1SjBG6B0CwsLCxMustNw/KtFdvR6fWJiYnFxMYvF6tWrl0AgePr0ae/evcvKynAct7CwsLS0lEgktra2jo6OAwYMGDt2LHqWwWDgGDXwv9i/W2Sn8ZLL5deuXXN3dw8LC0Mt9+/f79ev35YtW6ZOnYpaXrx4cfDgwenTp9OpShRFEQSh0WhgkR3mIjsSiQSSr2CRnVcX2eHz+bDIDiyy89pFdqysrBrp3qipqbl9+7a5uXmHDh2Y1dKKiooqKipatGgBJdQQOCSAOsflcmvNXA4ODs7KyiovL8/MzMzIyJBKpVlZWYWFhY8fP3ZyckLB986dOzds2ODt7T179mzUkpGRUVVVZW9vb2dn9+ZD+KNHjy5fvkySZM+ePV97kQu8mbm5+QcffMBsCQsLu3PnDjPdpaSkJCYm5uOPP6ZbLl68+MMPP+zYsYPOOJfJZBqNxt7e/g0HEplMdvbs2eTkZC8vr8GDB//b6zAAAAAaCAsLi4EDB77a7mRkii1qoCD4BibAZrPFRh4eHihRDOWElZaW0iOmbdq0+eijjzIyMvR6PWrZsmVLdHS0lZGdnZ2np6evry+a5enr68vn89HDoqOj582bV1paip6yefPmSZMmmeiDNh0CgQCVu6F16tTp6tWrjo6OdAvKO2JeGtq3b190dPTNmzfpk6WcnBwWi0VPEpXL5VOmTDl16hS62adPn/3797u6utbLZwIAAABMANJO3haknbxz2sl/QRAEjuMoOzkuLu7u3btZWVnZ2dmlpaVVVVU1NTVVVVUEQdy6datTp05qtXr37t3ff/99SUkJ/QpOTk7Jycl1Wsej6aWdvBu0wrBWq6XTTk6ePBkXF/fdd9+hvxqKooYPH56TkxMXF4cuPu7atWv27NnMF1m1atXXX3+NNQmQdsIEaSc0SDthgrSTppR2At4SHBJAg8ZMHW5vhP6vVqvLjUpLSysrKwMDA1G3derUKWbkjVLNli9fPmbMmKCgILFYDAlndQetvqTVaumW4UbMx3z22WfMQit3796t9SJ79+4NCQmJiIiA8W8AAABNEgTfoFESCoWvlri2tbX97bff2rRpU1lZSTfiOP7bb7/9+OOPIpEoJCQkMjKydevWKFPFxcXFFNvelL35ShqO42hhJtqrtVMiIyPHjx//9ddfL1myBLWo1er09HRvb29U3RIAAABo1GCJO9B04Dju4eExc+ZMZuOnn3566dKlEydOLFy40MrK6ujRo1OnTu3fv/+WLVvQA3Jych49eqRUKg0Gg4k2vPmaMmUKs3iiq6vrggULzpw5w5zH+fjx4379+sXHx9MtBEGgdVXrfXsBAACAuh/5VigUIpGILglcWVkpEokEAsF/fmsA6sSKFStcXFxOnTpFUdSgQYNmzJghEAgiIyOHDx9OkqRarc7Jyblz546Xlxd6/K+//vr1119bWlru2rVr9OjRGIYVFBRgGAbj4vXAy8vrzJkzmzdvzszMdHJymj9/frt27Wo9xsXFZfHixcySsXv37j1s5OfnV++bDAAAANTZhEuVSlVWVvbDDz9MnTrV3d0drYq3evXqUaNGdejQAWsw6mcATKfTyeVyc3NzmHCJJlzq9XqRSIQ1PGieil6vpyiKnuf32vVfUHtKSkpcXFxiYuLIkSM7depEkuS0adNQdB4cHNy+ffs2bdo4Ozvz+Xx0Cvrq700mk7HZbEiKQDtHpVIJBIK3r/ONvheVSoV+Tq/uXvqLo++6dOnSlStXFi1aRNeu+v777wsLC1evXk1Pe0XPMu3ouNrIwsICJlzChEsmgiBqamr4fH7D7ELrGVocQCQSwRRDDMOUSqVOp7O0tKyfvQH7vMEF3wRBrF+//uTJk1Kp1NfX18zMDK0OyuPxDh48+A7L7NURvV7PnOBVdwiCQB+fuS5gs0UaNeSQ4u1jLy6Xy+FwtFotSZLoox0/fvzatWvp6elFRUWlpaU6nc7FxSUsLKxz585jxoyxtbUlSZLP5+M4jlYC0mq1OI7DWRmi1+vZbPa//TNhsVho/7/Dl8tms5cvX15UVPTjjz/y+XyKokiS3L9/v729/YcffkgQBD0Z1GAwvP27/Hdo2VfoNBCCICiKasidRr0hSVKn06EVxEy9LaZHkiRBEA1/MbX6odfrCYKon+QCHMeFQiH0Tg1u5Lu4uPjZs2dnzpwZMmSIpaUlRVFcLtfZCGswKKN6eCOdTqdQKMzMzCDGauAj3/8d6oxUKlVeXl5ubq5UKn306FFcXByHwzlz5oynp2deXt7ChQtbtGjRo0ePzp07q1QqDofDHPmuzwivsY98vy/02Dm6FjFo0CAPD4/ff/8dfRcymezatWstW7ak1+OkDzl192Wp1WqNRmNubg4xFox8MxEEIZfLeTxeU+1C/xUY+WZSKpV6vV4ikdTP3oDI21TedEhwNGrduvUff/zx9OnTOXPm5OXlsdnsBhV840b19htlGdXD2zVwLBaLLr/dVIlEohZGvXv3xjBMq9Xq9XpUl7e6ujo7O/vmzZvPnj3r2rUrh8N5/vz5o0eP/Pz8PDw83N3dm+0xlaIo9MOo/98G8+TH0tLyypUraMwbbUlxcfHSpUunT5++ePFi9Jjq6uqMjAxfX9+6qwEPnQYTi8WiKAp2BTMRDvYGs9OA4Ju+rAd7o8n7h/EYrVa7atWqoqKiysrK7OzsvLy87du3//HHH3Z2dvW1hQA0CHwj9P+QkJDY2NjMzEw2m83hcNhsdnx8/IIFCzgcjqOjo5OTk5+fX5s2bSIjI1u2bAkpv/UPx/Fa5z/u7u6//vorszZlfHz8+PHjjx8/Tlc/pFcIgpAIAABA3fmHYwxaTXDbtm2hoaEYhs2aNcva2jopKakOtwiAxsDCwqJ169ZoxXWtVjthwoTs7Ozjx4+PGzfOzs7uyZMn69at69q1q7OzM712empqqlQqhYKGJiEWi9u3b8+sYBMcHLxq1Sq0PBOyZ88ePz+/iooKugWNnQMAADAJrVZbUFCg0+nq7i30en1hYaFGo6mj11coFEVFRbUSpP8h+OZwOCg7DU2UqaioqKmpabaX1AH4Ozwez8PDY+jQoevWrTt37tyNGzcuXrx4/PjxZcuWBQQEoB5kyZIlPXr06N27t1QqRRdbtVotxOKm4uLiMmPGDHt7e7olJCRk/PjxdL0UDMPWr18/efJkvV7/bm8BeQUAAPBfpKWlzZs3Lzc3t+7eorCw8LPPPnvx4kUdvf61a9eWLFlS6zjyD1fDvby82rZtO2PGjJqamqKiou+//z4wMLBVq1Z1tIkANFK1TmrtjCIiIj766CPUwmKxJk+e7OLikp2dbWFhgWFYZmbmmDFjhEJh+/btW7du7eHh4enpSdfO+ztqtVogEEA6YF3oZkTfpChKp9MJhUJ6bxsMhu+//97f35/+Wv+RUqm0srKqm+0FAIAmTq1WS6VSZlE7jUbz9tVgXltkiSAIZkkAnU6XkZGh0+nQ9INar/D2b6fT6WoV5CBJksViKRSKzMzMWkHCPwTfqIZXeHj49evXtVrtB0awwg4A/xaXyx1qRLeYmZn17dv3xo0bu3btksvlEonE0dHRzc0tJCSkd+/e/fv3r/UKZWVlW7ZsuX//vlAonDJlyogRI+r9QzQvOI5/++23JEnS3TFJkqdOnerWrRsdfGu12n379vXs2RNd32C6cOHC7t27KyoqWrVqtXTpUnd393r/BAAA0LjhxiqxqBO+d+/ejh07Kioq3NzcPv/8c5Iko6OjFy1aJJFITp069eTJk+XLl/P5/P3791tZWQ0ZMmTfvn0xMTEYhvXr1w8tfb1nz57q6ur4+PhevXp99tln6MokqhR84sSJDRs2kCQ5ZcoUdKR++vTp9u3b8/PzbWxsPvvss/bt2z958iQmJmb+/PkSiSQ7O3v//v2zZ8/Oycm5fv06SZK3bt0yMzNbuHBhx44dCYL49ddfo6OjraysLCwsXg2b/3keWE1NDSq4RpJkRUWFSqViXpYFALwbBweHVatWGQwGtVqdkpKSYJSamnr8+HG5XN6vXz8cx+Pi4i5evBgWFhYYGLhw4cJz586h5167dk2pVE6aNMnUH6LpY+aN8Hi827dvM4dMiouLv/nmG7FYTAffVVVVL168yM3N/fTTTxUKBTpgJCYmnj592tHR0RSfAAAA6hZBEGfOnCkpKREIBIMGDbK3tzcYDEePHkV9IBNFUba2tgMHDhSLxdXV1UePHu3Tp4+Pj8+bX5/L5SYlJU2fPr13797jx4+Pjo6ePn36ihUrzp0717t3786dO//yyy83btyYOHGio6Pjr7/+Onv27D179hw+fBgVQti6datOp5szZ05MTExFRcWiRYvc3Nzovh3H8crKylu3bn322Wfp6elLlixxdHR0d3efOnVqSEjIggULLl269Omnn0ZHRxcVFZ05c+bTTz+VSCRlZWUnT54cP358VlbW6tWrp0yZMmvWrF9++WXFihUxMTHnz59ft27dggULzMzMVq9ebWNjUysF8R+Cb5VK9fnnn1dUVIwaNYrNZsfExMTFxf3000/oujkA4D/icDjm5uZtjdDMj8zMTDqx5PHjx2vWrCEIYvjw4VeuXKGfpdPp1q9fP2zYMLFYDKVU6lOtMtUODg4XLlzw9PSkW27cuDF37lySJJlHnbi4uLNnz37yySf1u7EAAFAf0LKMDx8+tLKyCg4Otre31+v1X375ZX5+/qsPbtmyZceOHcVicUlJyaxZs44dO/bm4BvH/2dFmrNnzzo5OX3//fdcLjcyMnLIkCHp6enBwcEPHz4MCAioqalxcnJKSkqqqanBMMzf3//HH3/s379/ZGQkRVH9+/c/ceLE6NGjBQLBsGHDxo8fz3x9tP7Al19+OWzYMFQa4ezZs97e3lwud+PGjba2tr169RowYMBff/0VGhqK1tdD4zLoSE1RlL+//7Jly5ydnc3MzJYsWYJi9CFDhsyYMQPDsPLy8uPHj9daUOIfDtuZmZnFxcUHDx5Etb0HDRo0bty4p0+fdunS5V9+NQCAf8blclu0aEHfnDZt2siRI589e3b58uVaf7qpqanBwcG+vr4+/6tnz542NjZofcf6X+OmeRIIBBEREcyW9u3bb9myZeXKlUVFRcz29PR0+v9xcXEKhaJLly508UoAAGi8uFzuuXPntFotm822tbVFfeODBw9eLRhFURSPx0OP8fHxycvLs7Gx+cfXJ0kyLy/P39+fy+ViGGZlZeXi4lJZWdmpU6e7d+96enq6uroGBQUlJCTk5OT4+/tLJJLCwsLTp0/HxsaiQ6ezs7NOp2Oz2a++HUVREomEPgHw8vIqKioiSdLd3R3N2OFyuX5+frm5uS1btnzt5llbW6NxGbRgtlqtLi4ujoqKQvf6+vqKxeJ/l/NtY2NjYWGRnp7u4ODAYrEKCws5HI6lpaXBYGCz2TDrC4A6xeVybW1te/ToYWtru2XLFuZdrVu3bteuXWZm5rVr1w4ePGhubn758mUbGxutVjtq1Ci9Xt+2bds5c+agPk6pVHI4HC6XC8U36pqTk9PIkSMvX76ckpJCN+I43qlTJ/rmli1bUlJSbt26RQff27Zt4/P5n376KXPdzXpbQQwAAP4LHMeZlaNQyz/WD+BwOK6urm/5+hYWFnTNE4PBIJfLBQJBp06djh8/fvr06eDg4ICAgN27d3O53I8++sjc3FwgEMyfP/+DDz4gSZIwMjMzY87hYb64TqdTq9XoZnV1tbm5uZWVlVwuR7Euyif09PREz0UtBEHodDr61VBsjf7lcDgikaiqqgrdpVar9Xp9rff9hyOxpaUlSZIffvjh8OHDR48e3bdv3wcPHixZsmTYsGEPHjx4m10GAPjvQkND0VQSdNPPz2/37t179uy5cOHCy5cvS0tL4+LigoODUY/AYrGKiooePHiAaqNKpVJ3d/fWrVsPHTp0wYIFO3fuPH369J07d5KTk+neAbxfX3zxRVhYGPo/juOzZs3q06cPfe+WLVuOHj1KJ+8RBPHgwYN79+7RkbdcLp8xY8bWrVvppxgMhuLiYpVKVb+fAwAATIw01gzp2rVrQkLClStXZDLZiRMnpFJpp06d/P39eTze+fPnUepmRkbGixcvOnfubG1t3apVq5iYGI1Gw+Fwdu7ciRI4Xy1NhrroioqK3377rbS09ObNm3fu3OnWrVuPHj2kUunJkydramouXbqUkJDQuXNnGxubqqqqxMTE8vLyP//8s6KiAqWdMF+TJEmhUBgVFXXq1Knk5OS8vLzff/9dqVTWCr7/YeRbKBQuWrRo4sSJBoOBoqjRo0fjOG4wGHAch8n7ANSn5cuXd+jQ4f79+xKJZMCAAf7+/ugUnM1mCwQC+lKaWCw+deqUXC5XqVRo2NvCwmLevHnp6elSqTQuLq6srAzDMHNzcwsLi2HDhu3YsQPDsKSkpIcPH3p7ewcGBsL6tf+dn5/fGaP8/PzIyMhBgwYxZ7u7GtE3WSzW9u3bmRXfDQYDGlahW3Jzc1GJm9mzZ6MWhUJRUFDg6uoqFovr62MBAEC9YrFYIpGIoqgePXqMHj167ty5Tk5OZWVlc+bMiYyMZLPZ7du3z8jIaNGihaOjY1BQEIvFcnd3x3F8+fLl8+bNGzJkiFAoVCgU69atE4lEQqGwVjVAFHy7uLgkJycPHz68rKxs4MCB/fv35/F4M2fOXLNmzb59+4qLi8ePH9+zZ0+CIDp37jxnzhwnJycbGxt0FOZyuSKRiB4URyvhTJky5cmTJyNGjLCyssJx/NXZ9v8Ts9dqysnJSUtLKy8vNxgMEonE19c3KCiojndvI6DT6WpqaiwsLF795pohvV6v0+ngqI9UV1ez2eyGXwVIr9cbDIbKysrs7GypVJqSkuLj44NmAe7du/fzzz+Xy+ULFy7cvHkzhmGnTp16/Pixj4+Po6Ojra2tnZ2djY3NP35GiqKUSqVQKISkczQEgopIvsNzCYJgrtFTVFS0ffv2bt260TUob9261b9//717944dOxa15Ofnx8XFRUVFoZOuhkaj0aCJTabeENMjCEImk/H5fOhC0d5Qq9VisRiSrNBJtU6nQxGbqbelQSBJUqvV8vl81BlmZGSg5G80ERENVej1etSxaLVaVDcQ3aXT6Z4+farRaEJCQiwtLVEvxGazUeI48y10Oh1JkklJSWKxmJnYnZ+fL5VKPY3ot3v+/DlBEGh9ay6XS5KkXq9Hky/RoAnaWoIgnj17xmKxAgMDSZKsVW3w/wu+y8vL169ff+HChaysLJT+guO4nZ1dSEjI0qVLe/Xq9ZY7S6VSPX/+nMVihYSEvLYouFQqLSgo8PHxqZXuQ1FURUWFpaUlKuCg1+tfvHihUCgCAgJMfjiB4JsJgu/GGHy/QUlJSVpaWllZmY+PD1pFa/ny5Zs2bdLpdBwOR2JkaWlpZ2fn6ek5f/58VFlPpVKhno7FYtGHilcXGmi21EYWFhZ1UZGmtLT06tWrnTt3pi9CRkdHjxo16uHDh+Hh4ajl5s2bf/3114oVK+gUF3SF1CSp/xB80yD4ZoLgmwmC72bi/4LvmpqapUuXlpWVde/evVWrVnFxcSkpKZMnT37x4sXt27dfvny5YcOGnj17/uMrlpSU/Pjjj1qtliRJMzOz2bNnM+NmgiBiYmJiY2Pt7OzKyso++uij7t270/deuHAhJiZmxYoVzs7ONTU1e/bsKSgoMDc3l8lkU6ZMoXMoTQKCbyYIvptY8P0qvV6v0WiKi4ulUmlmZqZUKs3KyiopKZHL5QcPHmzbtm1NTc3gwYMzMjJCQ0P37dvn7Oys0WieP39uMBjc3NwcHR3fPuJUKpV5eXmOjo5ocKLJUKvVKpVKIpHUTznImpqatLS0li1b0gHuzz//vHHjxoSEBHr0fefOnQkJCfv27aPHfioqKths9r/a83l5eXq93tvb+19tHgTfNAi+mSD4ZoLgu5n4v0PCs2fPvLy8du3ahW4WFRVVVlZ2NJo2bdqJEyfi4uJ69Ojxjz+IS5cusdnslStXEgSxcePGq1evjho1ir43Ozs7NjZ28uTJ4eHh58+fP3v2bOvWrVExl7S0tNOnT9OzR588eZKXl/fll1/a2dkdOHDgr7/+CgkJgZLGANQPrpG5ubmfnx/dqNVqKysrUX45h8MZN25cXFxcSUkJiqjy8vKmTJlSUlJibW1tY2Pj4ODg5eXl4+Pj6+vr5+fn5eX12jf666+/Nm7ciILvWbNm1arACt6ehYVFrbqHo0eP7tOnD/O0UKfTabVaesyFIIgvvvhCr9cfOHAAZQpRFCWVSq2srF57sbGmpuarr766du2aXq8PDw/fsGGDm5tb3X8yAABoUv4vlmWz2XK5XKlUonNxZpUriqIqKytrZcm8llarTU5O7tq1K0o5b9u2bUJCgl6vp5+bnp5ubm4eGhqK43i7du2uXr2alZVlZWVVUVFx/PjxiIiIlJQUVJQRXRsViUQcDkcsFtcqclz/0N6Ak1EE9kPzxOfz6epRIpFomhGaio6q7K1ZswadNufl5eXk5Dx9+lQmk6FqrNevX+dwOM+fPz98+LC3t3d4eHhkZOSjR4/GjRuHlkXIzs5+8uSJra0tndbcNJjwj8XMiNmyYMECZtoJi8VCRXLoFp1O17dv3x49euzbtw+1EARx+vTp0NBQX1/fL774Ys+ePag9LS1NJpMdP34c9fb/CJUFeK+fr7GC/pMJjq2vgr3RjILvVq1aRUdHDxo0qEuXLsHBwU+ePMnMzIyOjk5KSnrw4IFSqdyxY8c//iDUarVGo6GvYEokErVardVq6eC7qqrKwsICDbHw+XyBQFBdXY1h2PHjx93c3Nq1a/fy5Uv0yIiIiMTExNWrV1tZWZWWlk6ePPm1w97o4jhW91D0r1KpoFIy2huIqTekQSAIgqIouVyONTO1SpziON6vX78+ffrweDyDwVBRUVFWVlZeXl5SUmJubq5SqXg8Xlpa2u7du5VK5cSJE8PCwg4cOIAib0Sj0WzatMnHxwctxksvJsDj8XAcJ0kSlV3CGtUPQ6FQNJxOA+1P5j5E5VPo9ThJkvzmm2+cnZ3R75nFYpWVlY0YMWLDhg2DBg26ePEi89ViY2MvX748YMAANJrOPECwjNAeQC2ozhezqEuzhTLv0RwvU2+L6VEUhco2m3pDGgTUxdXP3mCxWEKhsOH0Ts3K/4WzIpHou+++++GHH06fPr19+3Z0RDx9+rSXl1eHDh2WLFkSGBj4jy+HCpjThQ7YbDYqb04/gCAIDodDn+my2Wy9Xn/16lWlUjlt2rTCwkJ0oEVFV/Lz89EKQ1VVVcnJycHBwa/G3+h4jNUxdOBH2w/dJX3whuMoghaVhL2BToZJktRoNDiOm5mZWVhY+Pv7oyBMq9VqNJqoqChUTInD4VAU9Wqh8WvXrrVo0cLGxsbDw+Obb77p0aOHTqd79uyZSqWyt7d3cnJC/QP6F8V2zDKrDSo0RxvTwE9T6bAbwXF85MiRqIAAumlmZnbr1i1XV9fCwkJ6HQpEp9Oh+rV37tz54Ycfvvrqq1atWqHlJKqrqzUajZWVFfqi6aupzHC8OUM/Wug0ENgVNJIk621vsFgs+GM0lf8vlhWLxcuWLZs5c+bz58/Ly8spikLHzrcv6Y2W0NNqtegmWsyTOUNRKBSiuZjoeEwQhFKpvH37tk6n27hxY01NTXFx8Z49ez7++ONLly55enrOmTMHjcrv2bOnffv2r6aN8ni8t8mH+Y/QAkhyuVwsFtfD2zV8eqO3vNzc5MlkMjabXev6frOFSg3WGk3hcrmo8BGKwOgqUUOHDj169CjzkfPnz/f3909JSZFKpRYWFmKxmMfjrV+//u7du46OjgcPHoyMjDQYDNHR0Xw+39/f383NTSgUovU7mWENeiPTRr0ajUatVpuZmTW6wovMThvH8S5dumAYhipflZSU0HcFBQV16dKFy+Xy+Xw0mZJeQ+7YsWNHjhz5+eef/f390QE+MzOTIAgfHx9m0gv9fWHNCUmSNTU1PB4PulC0N9RqNV0puZlTKpV6vf7d6pP+W7CGrgm9JpHD0tISdbXvQCQS2djY5OXloZu5ubl2dnb0snwYhjk7O9+5c0epVEokkoqKCpVKhVbspDt0NpuNKnNpNBpra2t6k3Acf216Sb39eujRerhGg46asCtq/Qhhb6BA6l/tjY8++mjZsmV79uypqqoSi8UTJ05ct24dCtPpdYA5HM6iRYt69OhRVlZmZ2eHRk+3bNny8uVLlNLm7u6OpnV6eXkFBwdHRUUJBAKKogiCMG3U28Q6DTMzs82bN0+bNg0tb+zv779582Y0NNO9e/du3bqhj4n+dXNz69Chg4WFBd1Fr1279t69ezdu3KDXm7hx4waK7JvG/nl79KWA5vbB39BpMMuVNmd0pwF7o2l7z8VD0GpDx48fd3FxIQgiLi5uzJgxLBYrMTExLy+vX79+gYGBfD7/yJEjnTp1On/+vIODQ0hISJs2bdDTMzMzc3NzR40a5eDg0KZNm1OnTjkZnTt3ztbW1sPD4/1uLQDAtFgs1nfffffBBx/k5eU5ODi0b9+eHsCm4xIcx6OM6Gfx+fxDhw6VlJSUlpbm5ORkGMXExBQUFLRp0+bKlSsCgeDly5ezZ8+2sLDo16/fjBkzWCyWTqdDk7+5XC4c2N5NaGjon3/+iabRt2rViq42yMw2RPobMVsmTZrUr18/ekiFoqhvv/2WIIjr16+jFp1Ot23bNicnJ2bFG3QOBt8XAKApef+V+7p06aJQKC5duoRh2JAhQzp27IiupFRWVhIEYWlp+cknnxw/fvz33393cXH5+OOPmVkcAoGgbdu2qKVnz54kSd68eZMgCFdX1wkTJsAVOgCaHlT4qF27dv/qKaGhocwWlKMsN0ITvtGSY9nZ2U+fPkVx/NGjR+fNm+fp6enr6+vt7e3j4+Ps7IxW7nR2doasobfk4OAwaNCgd3hi586da51THThwQKVS0VG7Wq2+ffu2j48PHXwXFRUtXbq0Z8+eEyZMQC06na6qqsrS0pJ5QRUAABqXuqr9hNK+/65/RGtQv83RDg1WNYSVCGCRHSZYZKfJL7Lzbhra8vJVVVUEQaCS1U+ePImOjkYLBmVmZqI6S2jxzq+++mratGkYhl28eLGkpMTf379ly5b0qpDvDM09hVGDt1xkhyTJ0tJSNpttZ2eHWvLz8xctWtSrVy/07WAY9vTp0+nTpy9ZsmT48OGoRSaTFRcXe3p6NqJwHBbZYYJFdphgkZ1moq7WrHlzP4hm0L/N6/CM3t92AQCaEbSAF9LaCFUS0Ov1ZWVlWVlZKBD39/dHjzl16tSRI0fUavWpU6eGDRtGkuS2bduqq6t9fHwcHR1t/9fbxEwkSd6/f7+wsLB169bMhYrA32GxWHQ6OOLi4vL7778zoxBzc/P27du7uLjQLTdv3vzggw9u3LjRrVs31JKZmfno0aP+/fv/q6sZxcXFSUlJQqGwTZs2sAwnAKBOwaoHbwtGvplg5JsJRr4b7Mj3v5VtVFVV1bFjRwcHB7VaPWnSpFOnThkMBi6XK5FILI3s7Oz8/PzWrl0rFovRuB0qt0J/ao1GM3/+/JMnT1ZUVPj5+S1fvpxOnGi26mh5+fz8/KtXrw4ePBgtvIph2A8//PDZZ58VFhbSC0L9+eefDx48+Oabb+hitagMDp0Dc+PGjUWLFj179kwoFEZFRe3Zs4cuyFMXYOSbCUa+mWDku5mA4PttQfDNBME3EwTfTSb4roWiKL1er1arCwsL0bROqVSak5NTXFzMZrNjY2NFIlFiYuIHH3zAZrN79uy5e/duLpdbWVm5cuXKnTt30q9jZWV14cKFf5XX3vTUUfD9qqqqKqlU2qZNG/pH+N133509e/b69euokA5FUatWrZLJZBs3bmSz2Tk5Of37909OTqZfYdy4cQcPHnztsm7vBQTfTBB8M0HwbSoURZWWlgqFwv+ecPhaBEGUlpai+rl1mHYCAACNHY7jKPNNIpEwVxlTq9VyuRzFkU5OTnPnzk1KSqLPvuLi4piRNwoH//jjj+DgYD6fX3chHUCsrKzatm3LbJkzZ87EiROZyZA6nQ6tyokmAzAjbwzDzp49m5iYGBYWhtbprMdtBwCYhlqtXrlyZZcuXcaOHVsXry+TyRYvXjxu3Lh+/fr9z2W3ungPAABowoRCob29PRqacnR0XLx48S+//LJ582ZUqSkkJCQ4OLjWU7Zt2+bi4vLzzz+jm8XFxfR6CKCuSSQSV1dXeigRx/FVq1b98MMP6ESIbcR8vFAoVKvVUVFR0dHRJtpkAMA/Yy6gTnvt+qCvXSiGmfpBkmR2djZa3P3VlzUYYW8HLQ9c6+ksFis7O1smk6EWGIMBAID3gB4idXNzmzp16oIFC+i7nJ2dFyxYUFhY6OPjg1o2btwYExPj4eGxdu1alI6CemcYZ60fzGg7IiIiLCzs4cOHdMuwYcNatWrl7OzMvACdnJy8evXqZcuWhYSE1Pv2AtBMZWZm/vLLL9OmTXN1da2oqNi2bdukSZOsra337NljZmZ25coVHMcnT548dOjQnJycI0eOCASC2NhYgUAwb948tDrE48ePt27dWlpa6urqumDBguDg4Pv3758/f760tLSmpmbNmjWoW8ZxnM/nx8fHx8XFFRcXDxo0aObMmRwOR6VS/fTTT6h8dv/+/WfMmIHj+I4dO1q3bt2zZ0+Kon7++WcbG5shQ4Zs27bN3Nz81q1b5eXlffv2nTVrllAoTE5O3rhxY3FxcXBwsFarpa98QkcPAADv2Zw5c7766isfHx+UAvHTTz99/vnnW7Zs6dOnD3pAp06dOnfurNfr0cw/nU43YsSIIUOGfPfdd2fOnElMTKyoqDD1h2guHB0df/rpp+7du9vZ2bm5uU2ePHnNmjUSiSQ6OnrgwIH0w0pKStLT0xUKBd3y8OHDQ4cOVVVVmWjDAWgoCILYt29fTEwM3WIwGH744YerV6/SLTqdbv369ffu3aNbZDLZunXrUlJS3vDKZWVlp06dQn9lCoXi5MmTpaWlKpVq3759v/zyy4gRI1q3br148eJ79+4plcrNmzefPXt24sSJHh4ec+bMefnyZVZW1uzZs9H1SbFYPGfOnIqKioyMjC1btjg6Og4fPpx5gq3T6WJjYyMjIz/88MOdO3ceOnQIw7ANGzYcOHBgrNH+/fs3bdqk1+svXrz48uVL9Kxr1649fPiQoqgjR47s27dv4MCBQ4YM2b59e2xsbFVV1Zw5c/R6/cyZM8vKyl6+fEmf9sPINwAAvGccDmf16tWjRo0qKCho3bq1vb19rQd8aERfx2SxWD4+PteuXfv222/1er2dnZ2jo6OXl1erVq26GsE87zrVpk2bv/76KyUlRSAQtGzZEjXWmvHWtWvXK1euMCeMnj9//ueff27fvj1d0fLRo0c4jrdq1appTDgG4C0RBPHjjz8GBQV98MEHqEWv12/YsGHkyJG9evVCLTqd7uuvv161ahVaexFNhvnqq6/8/PwCAgL+7pVZLBafz0eXBHEcFwgE6P9mZmZLliwZOXIkhmFpaWlnzpwZP368vb39ypUro6Kihg0b9uzZs8uXL/N4PIIgRo8ebWZmNnLkyOvXr8fGxrLZ7KCgoNmzZzN7ZpSCMnny5NmzZ6Og/+zZs3369Dl37tznn3+OVv4iCGLXrl1jxowRCAT0GDaPx+NyuRRFcbncqVOnjh07lqKoc+fOZWVlcTicysrKw4cPu7i4hIaGPn78mO7zIfgGAIA60aJFCw8PjzdUtKC7bw6Hs2nTJlRS6fnz5/Hx8YmJiVKpNC4u7s6dOx06dODxeBkZGefPn/fx8QkICKDXdQfvi1gsDg8Pf8MDWCwWWjyVtnjx4qlTpzKP3xs2bMjMzLx58yb9paelpZmZmdUqXAiFLEATw+Px7t69y8yaEwqFaWlpzPnlYrG4qqqKOY7g4eEhk8lQDaJ/haIoS0tLehK8l5dXTk6ORqNxcnLy9PRE2+Pl5ZWdnS0QCPLy8ubOnUtvFZvNRqutM5dXR7hcLv2a/v7+V69ezc3N1ev1LVq0QI0BAQFarbaysvK1f8J8Ph8tEEYQhEgkIkmyoKBAIpGgKqg2Njbu7u50ljkE3wAAUCf0er1Wq2WOkbwZj8eztbXtboRhmFwuz8rKIkkSRXKJiYmfffYZhmGzZs1C1VRevHhRVlYWEhJiYWHx6oEE1DWhUMhc7gfDsGXLlimVSjrylsvln376aWBg4K5du5jFK02xsQDUrVeX8q017vDq6or42623SFEUCuspitLpdKhRr9fTcyhRGWgOh6PRaOgHyGQyNzc3kiRbtWp16NAhFouF47hGo7Gysjp16tRr34gkSTqvrKqqis/nW1paslgsuVxOvyb6w6coCsXfOI7TdZNw/P+Kd6P/mJubq9VqnU4nEAhIklSpVPR7Qc43AAA0RObm5qGhoa1bt0Y3P/jgg5ycnJMnT44YMQK1REdHd+/e3dnZuWPHjjNnzty/f//169czMzMhvDOVsLCwzp070zfZbPakSZP69u1Lt6Smpnbr1u3w4cNvXgQaAIBYWloqFIrExMSqqqrTp0+XlJSgMLqiouLQoUPFxcV37ty5fft2r169hEJhdnb20aNHy8rKLly48OLFix49enTt2lUqlSYkJJiZmb148eKTTz5JTU1ls9mvLnGD47hOpzt27FhOTk56evoff/wRGRnZokULf3///fv3Fxrt37/f39/f3d1dIBA8evSooqLi9u3bDx8+ROcGJEkyg2+SJCMiIrRa7a+//lpeXn769OknT55AzjcAADQmXC7XzYhumTBhQmhoaHJy8oMHD2JiYvbs2SMUCh0cHNzc3LZt29amTRsMw7RaLZvNhuLiJiESiSZOnMhsMTMza9++vaurK0mS6ICdkJCwbt26lStXtmrVynRbCkAD5enpOWjQoBUrVuzdu9fa2jogIAD94YhEooSEhI8++qi6urpv374DBw5MT083Nzc/f/781atXy8vLJ0yY0L59e4qiJk+evGjRIldX16Kiol69egUGBqalpYlEoldTR6ytrYuKiiZMmFBVVeXr6zt79mwul/vVV1/Nmzdv2LBhGIZZWFhs2bLFzMxs7NixS5YsGTp0qEQioXMLxWIxfQVSKBTiOO7t7b148eI1a9b88ccfXC7X3d2dfgCscPm29Hq9TCaTSCRweRdNZNZqtbA8GwIrXDIpFAqRSAQl89CqDSqVytLSsq6n31EUpdVqc3JyHj16FBcXl56evnPnTh8fn5KSktGjRxsMhs6dO69YseIdcisb4wqXDR9JktXV1WiFy5s3by5ZsuTgwYNBQUHo3vv379+6dWvevHnNZF+hy/Fvk37QfFa4tLa2NvWGNCAURb18+VIul4eGhqKhhOzs7DFjxqxatcrOzk4kEqH5momJidOmTduxYweHw7G2tqbrumIYlpOTI5VK3d3d/fz8UACj1+sFAgEz/ka9KIvFSk1N1Wq1oaGhdHq6Wq1+9uwZi8UKDQ2lr1kVFhZKpdIWLVpYW1tTFMXj8dRqNZfLRSMdGo2GxWKhVygsLMzKygoICDAzM6OHQhp98I2yKuvhjUiS1Ol0XC4XprGjvUEQBJyHIFqtFi2FaOoNMT2KogwGA4fDgSll9LoMPB6vrk9F0N5msVioZ9doNKiCYVVV1U8//RQbGyuTya5du2ZlZfX8+fO1a9c6OzuHh4eHhYW5uLjwjbhcrk6ne/tVJN4BenEYg8dxHB1NWCwWl8vFcVypVNI1HFgs1qZNm7Zs2ZKUlGRlZYUO0HFxcTqdrkuXLvSfFY/HI0myTr+vegOdBhMqP8rj8ephb+A4LhQKG+NAiVQqHTFixLZt27p160Y3Pn78eNKkSUeOHAkNDcUavEYffJNG9fBGer0ezaSBiJOOKkw7kNZwyOVyFosF1wHQcVSj0fB4PDhHRWdlGo0GjXbU5/vS835wHGexWCiq5vP5bDb78ePHCxYsyM3NLSkp0Wg09vb2bdu2bdOmTXBwcGhoqK+vLx0fo5zFVw8QzOj5XwV/aJQEcp1RMQSlUsnlctHYNovFYh7F9Hq9XC63srJC4RdFURMmTMjJyYmNjUVHH5Ikk5KS0PVu+lksFosyYr4RjuP0b6/ejpX/FkmSGo0GXaY39baYnkqlMhgM5ubm9RN8o/xprLGpqKj49ddfBw8ezBzeLigoiI6OHjVqVK3iQg1Tow++6w2qAmZhYQEDnOjwoNPpINxEIO2ERlGUUqlE5ZxMvS0NJe1EIpE0qOFevV6fa5Senp6YmBgfH//y5UutVrty5cpvvvkGw7AHDx7k5eW1a9fOwcHh1bGG5OTkU6dOaTSaQYMGobU53xKkndAIgpDJZCjt5G0en5SUZDAYWrVqheIkjUYTERExcODADRs20I9RKpV8Pr/WL81gMJw5cyY+Pt7Pz2/EiBF0PfIGhSAItVotFosbYxRYR2kn9KkXaKoa0CEBAABAXeNyuT5GqKChTqeTy+WPHz+mq+adPHly06ZNJEkeO3ZsxIgRFEW9ePGCoihfX99r165NnDgRrb65efPmbdu2TZ8+3dQfqOkLDg5m3uTxeEuXLkX1jJHc3Nw2bdrs3Llz1KhRdGNlZeWCBQuOHDmCBrwPHDhw4sQJV1fX+t12AMBrQPANAADNF4/Hs7Gx6d27N90yY8aMqKio1NRUlDqp1+vnzZv38uVLLy+vzMxMet17tVq9du3a7t27ozlMoN6wWKxx48YxW/h8/pQpU5iX4O/fvz9v3rzHjx/TqSbx8fGbN2/esmULDKkCYHIQfAMAAPg/Xkb9+/dHN7lc7pIlS+7du3f9+vWysjLmI3NychITE/38/FQqVXx8vEQicXZ2dnBweG14BzFf3XFwcPj++++ZLUqlsqCgoFaS959//jlnzhw6Rq+srMQwDDIcAKh/jW+WKwAAgHqD43i/fv1WrVp17Ngx5tgqqmXr4OCAYVhKSsrQoUP79u27ZMkSNAszLS1tx44dZ8+eTUlJqamp0Wq1BEE0xroKjVSvXr2++OKLWo0SiYQ5S37NmjVTp06tqamhW6RSaWFhYT1uJgDNFIx8AwAA+GdOTk6TJ09evnw53TJx4sSOHTtiGObt7X348OGXL1/a2tqiOZovXrxYvXp1dXU1qqfr5eXl4eHh4+Pj7+/frVs3tKCMRqPh8/kw7FpHhgwZcujQoSdPnqCb1tbWW7dupTP7MQwLCgqysrKip8ASBDFp0iQejxcbG4u+FJ1Od+3aNVtb24iICBN9CACaJqh28rag2gkTVDthgmonNKh20vCrnfwXOp3u4MGD0dHRGo1m8ODBs2bNkkgkr31kdXV1RkZGWVlZcXExWuECKS8v//bbb1esWIFGXi9fvuzi4jJv3rwOHTpgGCaTydhsNqo7jjVd/7bayTtLSUnZvn07qnYyY8YMNMX2DVt17do1iqL69OmDWoqLi7t169a1a9eff/4ZtZSUlOzatatLly69evVCLai+IW70bhsJ1U6YoNpJM9FEDgkAAADqGo/H+/TTT6dOnUpR1JvjY0tLy/DwcGaLQqHQ6/VqtZou9Y0qoCcnJ6PMB4IgPv7447t37wYEBPj6+np7e/v4+Dg6Otra2trZ2bm5uTWZc5h6ExAQsHv3bq1W+zaLtrDZbOa8WwzDbG1t//zzT+YXXV1dfeXKFebw+bNnz1atWjV//vwuXbqgFrVaLZPJbG1t4fsC4O/A3wYAAIB/4d2CKq4Rs9T0fKPKykp6APiTTz5p2bJlRkZGUlLS2bNnFQoFm82WSCSOjo63bt2ysbGpqKg4ffq0ubl5ixYt6LrX4M3eeWEjDofTokULZouPj8/x48drLQUvl8uZMztv3ry5cOHCY8eO0RUSy8rKqqqqfH19IekfAASCbwAAAHXu1fUXEWtra/QfNpv9kRFaOF2v1xcVFWVkZKSnp5eXl6MAvbS0dMeOHSkpKd7e3omJiQKBIDk5edeuXS4uLp6ennZGtkaQH1gXOBwOc9gbw7DQ0NDz588z08xsbW07duxoY2NDtxw4cODLL78sKSmxtbVFLY8fPy4uLu7Xrx8djsN5FGhWIPgGAADQgLBYLJ6Rr1Hfvn3pu7y8vH7//feioiIOh4PC6+rq6rNnz+bm5pIkKRQKJUZWVlZ2dnYff/zx2LFj0ZLdaN4nl8t97eCrXq/fu3fvlStXxGLxhAkTmO8I3gzH8VpXQiIiIvbt28dsGTRokLOzs6WlJd2yc+fOW7duvXjxAs24JUly165dbDZ7+vTpdBROkuTfpZI/f/589+7dBQUFrVu3njVrFiq5A0AjAsE3AACAxkEgELQ0olvatWuXnJysVCrz8vKkUmlGRoZUKs3Ly8vNzS0qKkKP2bx587fffuvv77969erhw4djGPby5cvy8nIHBwc7OzuBQLBkyZIff/wRPfjUqVP79u1DUTt4L2p9ZRiGrV69uqKigs4mpyjqzp07AoFg2rRpdKGV6dOnt2vXbubMmfRjiouLbWxsnj9/PmLEiKysLFS5/Pr16zExMcx0JgAaPgi+AQAANFYsFktgZGNj07p1a7q9pqaGHuTu2bOnTqdLTk62sLBAYdyGDRtOnjxpbW1tZWUlEomePn1KP1Gj0WzZsqVPnz52dnam+EDNgosR+j9FUWw2e+vWrejbRI04jisUCo1GQz+lpqamV69eixcvfvDgAYq8kdu3bx89enTWrFn1/iEAaHjBd1VVFY7jzMtMTBqNRi6XW1lZvTpxByUF1rrShCoZ1dGmAgAAaGJQnI10NKKPIziOL1iwoHv37lKpNDMz8+nTp2q1mvncly9fJiQk3L17l81mBwcH9+vXz8LCgqIolUoF9VXrSK3UES6X+8cffzBzhCiK6tq1q6OjY3Jycq3nXrlyBQXf+fn5CQkJXbp0oZPLAWguwbdWqz19+nR8fDyO4x07dvzggw9qVaRKSkqKjo6Wy+UODg5jxozx8PCg70pNTb148eLYsWOZfzmPHj1KSEgYM2YMszMFAAAA3h5zBKe1Efr/w4cPe/XqJZPJmPe2atVqzZo1CQkJ3t7e7dq1s7CwKCwsHDlypJmZmZeXl4+Rt7e3h4eHUCjkcrlQ2P4/enWIrdbYnKWl5e7du1GqyY0bN5h30fXLb9y4MXPmzAsXLnTu3Bm1XL169f79+9OnT6eD+9cO8AFQz95/3Z9r167du3dv2rRpEyZMuHHjxq1bt5j3lpaWHjp0KDAwcPHixWKx+JdffqGvK1VXV+/fv//+/fs6nY5+fEFBwf79+xMTE5mNAAAAwHsRHh4+e/Zs+qalpeXSpUtdXFyuX79eXl4eGxuLEiT0er2jo6NMJrt+/frmzZvHjBkTFhbm4ODQokWLr7/+Gj03Pz//+vXrz549q6ysfMM7QuT3X8yaNSswMJC+2b9//ylTpqD/Dx48+NatW6GhofS9L1++vHz5slKppFu2bNkyd+5c5rlWeXl5dXV1fW0+AHUw8m0wGBISEjp37hwUFIQu9iUkJERFRdGjAqmpqQRBDB482NzcfNiwYRs3bszOzg4ICDAYDCdPnuRyufb29vSraTSa48eP29raajQa6K0AaPjoy/qm3pAGBPZGA4fj+DfffBMaGnrjxg0LC4vBgwejcVO+Eb2Ep6en58mTJ3U6XVlZWblRaWlpdnZ2WloaPdvv9OnTS5cutbCw+Oyzz5YuXYphWGxsbGpqKirbYm9vz+PxOBwOi8Xi8/kwWP5unUZwcHBMTEx0dHR2dnabNm0+/vhjuu64RCIJCwtjPnjy5Mkffviho6Mj3aLT6dBCqnTL4sWLSZI8ePAgynKhKCozM1MikZgwdwU6jSbvPQffKpWqurra2dkZ3XR2dn7y5IlWqxWJRKiluLjY2toa3ZRIJEKhsKSkJCAg4PLlyxUVFcOHD//111/RVSGSJGNiYthsdr9+/WJiYl5bIBYtiqbX67E6huO4wWBAgx/M1QSaJ7Q3DAYDczZMc4Z+ErA30HGLIAiNRgOraeA4jromrVYLewN1nmhv/F1nbkIsFuvjjz8eOnQoi8XicDhv+FtmsVgODg7Ozs4oPCJJEh0U0AhR7969zc3NpVJpYGCgVqtls9kXL17cu3evQqEgSdLc3NzLy8vb29vLy8vZ2Xnw4MHe3t5ocXWRSMTn8+mdQ0eB6E0b4B57752GWq1++4jT29t72bJler2ez+e/+UjE5/MdHR0JgkBHcAzDPvvsMzTFE31lBEH4+/vrjVB7eXn5kCFDgoKC/vjjD71ej6Z+3rt3z93dPSgoiCAIuhqmXq9HNxEcx//7N0UQBEVR/2pvvDMcx99m6VPQOEa+KYqiVzfgcrm1gmONRoPqeqKfL5fL1ev1GRkZCQkJ06dPRz879PRHjx6lpaV99tlneXl5zNesRa/XKxQKrL7UmpfTzNXDaU9jQZJkff4OGzj4YTCpVCpTb0IDotVqsYbq3YIn5rPs7e0//PBDFotFEIRcLmexWJMmTerZs2d1dXVxcXFOTk5WVlZ2dnZsbKxSqfTx8XF2dq6qqvriiy9yc3M9PDy++uorR0dHnU6nUCj4fD6Xy+VwOOhf1Mkwg8i33zw2m00aYU2r00Bh8Ts8q9b5zJw5c3Acl8vl6F6CIJYvX25hYSGXyw0GA5fLTU1NnT59+oQJE77++muNRsNms7Oyso4cOTJgwIDw8HAUf6M1pJiFydGe/7ffF8uImSdTd9CpJlyBaQrBN5vNZrFY9K+NIAj0+6MfwOPxUIBO/1iVSuWRI0cEAkFWVlZBQYFCoUhISAgICDh69Ki3t3dSUpJUKpXJZPHx8R06dKi1qi06r601obOO6HQ6pVIpFoth4TTUURoMBqFQaOoNaRBQUbNXf5zNEKoIIRAIoENHYw0ajcbMzOzd1mNvYtCwrkAgwJoNiqKsra2ZJa7pa4a5ublOTk5mZmYGg8HFxaWgoCA9Pd3GxsbKyio1NXXQoEHV1dX+/v4+Pj5eRg4ODra2tg4ODi4uLjiOs1gsOoh8Q1SNwu7i4mIrKyuhUMgcpm04CILQarVCodC0Q7DocEZRlKWl5ZgxY9CXhYLp1q1bX7p0SSKRiEQioVDIZrPT09NjY2O7du1qZWVFEASLxbp79+6GDRvWrl3bsmVLFOEolUqFQmFra0t3huhb+7tvAX2thYWFKpXK19f3zd/s+wIX5UzlPR8ShEKhSCSqqqpCNysrK83MzJi9rbW1dWJiosFgYLPZarVao9GIxWIOh5NhpNfrZTLZuXPnUOnWpKSkZ8+eabVahULx559/+vn5vRrf1Aru6w56F7ZRPbxdA4fGYGBXMDtN2Bvo0IV2BewN+sAGewNhsVjosj7WjHGM+Hy+h4cHl8vFcdzKymrTpk0kSVZWVtrY2OA47uDgsGTJkufPn2dmZj548OD48eNarZbL5Uokkp49ex45cgQFf2fPnvXx8QkODvb29v67t3v8+PG6desyMzNtbGxmzZo1dOhQrEFCx/GGlv9AnzOLRCLmJE5UD+evv/6ysbGhYwMU8DCXUD179ux33313/fp1eiZbSUmJRqNhVnhj0uv1O3bs+OOPP7RabZcuXZYtW8ZMVQdNzHsOvnk8XsuWLe/evRsREUGS5P379yMjIzkcTmlpqUKh8PT0DAgI+PPPP+/duxcREXHz5k0Wi9W2bduoqCh0ppienr579+7PP//czc2tZ8+e6DUfP34cHR29ZMkSWEIWAABA0xi8YF4PYbFY9PQ+Kyur6dOn0wnlWq02Ly8vMzMzNTXVxsYGxXZSqXTt2rWVlZWTJ09Ga7lfuXLlxIkTAQEBbm5udnZ29vb2VVVVEydOlEql6GXv3r17+vTpPn36mOhDNyk8Hs/d3Z3Z0qlTp5s3bzK/U3t7+/bt2zNHDNeuXXv06NGcnBx6lP3+/ft6vb5r1644jm/atOnLL79Ej0xMTJRKpTExMXDRrKl6/99rnz59MjMz165dS1GUq6sriqHv37//9OnTRYsWubq6Dhgw4NixYxcuXFCr1WPHjqXnkqMcEqFQyOPxUNpTrcaGdloMAAAA1BFUFIXP56Pl2QcPHkzf1alTp4sXL5aWlrq6uqKW3NzcmJiYvXv3YhgmFoutra0NBkNRURH9FJVKtXv37k6dOnGN4Hj6fuE4XitQ7mXEbBk7dmyXLl3ozFWSJL/99lutVnvt2jW1Wr1r1y7mgy9cuHDt2jU4WWqq3sPk3FcZDIbMzEwcx728vNDPEU0KpqdalpaWFhcXu7m50RWaEHSiXyvOJkkSTXowbWeh0+lqamosLCwg5xt9oTqdDhZ7Q6qrq9lstrm5uak3xPTQLA6UFmnqbTE9tVqtUqkkEgkMX6EMeIqiYKIIynKWyWR8Pv89dqEEQaCDVE5OjlQqzcjIOHny5NOnT2s9zMLCIiAgYM+ePWFhYejqNI7j9vb2tra2f7cidV1DpU7EYnEzPB/IyMjQarVBQUEVFRUuLi61piOvXbt22bJlpts6UIfq5JDA4XD8/f2ZLehUm75pb/R3J/qvNkK8CwAAAPwdNpstNHJwcIiMjMQwLDQ0tFaS97BhwwIDAwsKCtBELK1WO2nSpLKyMisrK4lEYmdn5+rq6uvr6+3tHRQU1LJlSzhprGs+Pj7oPxYWFh06dGCu3CmRSMaNG2e6TQN1C/60AAAAgKamb9++n3322e7du9H60P369du7dy9z4Rg+n79379709HSpVJqVlVVUVBQfH4/W3AgODr548aKtrW1GRsaGDRvc3d3btm3bp08fVL2eIIhmVbWmHnC53PXr13/yySdJSUkYhtnZ2a1du9bNzc3U2wUaVdpJkwRpJ0yQdsIEaSc0SDthgrQTJkg7qdO0k9fS6XTXr19PS0tzdHTs1atXrTzPV7eqvLy8rKyspKSEx+N16NCBw+EkJCSMHz8+PT29X79+Z8+exXH83Llzq1atsrOz8/T09PHxQSPlbm5ufD7/3YpGN+e0k1qys7PPnz+vVCq7d+8eERFh6s0BdQiC77cFwTcTBN9MEHzTIPhmguCbCYLv+g++/zs0EUupVOp0OlT57sqVK1u3bi0vL6+qqpLJZNXV1WilSTc3t5UrV6JMiWfPnlVUVDg4OHh4ePzjZ4Tgm8lgMGi12ob/wwD/ERwSAAAAAPCmiit0S28jlUpVblRWVlZaWpppRE/ZXL9+/enTp21tbfft29e3b1+Koo4cOaJQKAICAry9vW1tbXk8HofDQdE281+g0Wh0Op1IJIId0rRB8A0AAACAf0EkErkbvfbeL7/8sn///jk5OWgBII1GExMTc+XKlZqaGgzDLC0tfXx8vL29fY1Gjx6N4/i/HfHNzs4uLS1Fofx7/WQA1AdIO3lbkHbCBGknTJB2QoO0EyZIO2GCtJPGmHbyXpAkmZubW1hYWFFRUVBQkJWVJZVKMzMz09PT+Xy+VCo1MzPLyMiYO3cuRVHh4eHfffcdh8PRaDQKhUIsFnO53Fp/QevWrTt8+HB5ebmnp+fSpUuHDx+ONRUKhUKn01lZWcHId9MGhwQAAAAA1BUWi+VpRLdQFIUW75TL5ebm5gRBcLlcR0fHzMzMgoIC9JgbN24MHjzY3t4eTev08fHx9PR0cHBISEigV4IsLy+fNGmSv79/SEiIiT4cAO8CRr7fFox8M8HINxOMfNNg5JsJRr6ZYOS72Y58v82ES7QMu16vV6vV5ubmOI7n5uYeP348IyMjMzNTKpXm5uaimZ0Gg4EgCOYrdO7ceceOHa1bt0bpKAqFwtra2tzcnMPhoNWy2Ww2vWZ2Awcj380EHBIAAAAAYEoUReE4zlyPz93dfdGiRSg6NxgMarU6Ly8vPT3966+/fvnyJfO5CQkJKSkpKPhesWLFr7/+iipnW1lZWVpaWltbW1lZ2dnZLV26NDAwkCCIixcvqlQqa2vrtm3bWlhYUBRVWFjI5XLFYrFQKDRtmM7hcEiShMi7yYPgGwAAAAANFNuIz+dbWlqGhIQUFBTMmzePvpfL5R46dGjAgAHo5pw5c/r27VtdXV1VVVVZWVllVFFRUV5erlAo0OD6559/npKS4ubmdvbs2dDQULVaPXToUJVKJRAIxGKxSCSSSCTW1tYWFhZBQUEjR44UCoUqlUoqlUokEhsbGzRIT1EUSZIsFus9BsoEQSQmJspkss6dO6N3AU0VBN8AAAAAaBymTp2anZ195MiR0tJSHx+fzz//fNSoUfS9kUbMx5P/C42p8/n8c+fOlZWVabVaLy8vFNz36tWrsLCwurpaqVTKZLLi4mKNRqNSqUJDQ4cOHSoUCp88edKlSxeSJPv27Xv+/HkWi3Xr1q21a9eiMN3Kysra2trSSCAQ+Pv7t2jRAsMwmUymUqlEIpFYLP7H3LOcnJy5c+deunSJoqjAwMDt27dHRUXV2V4EJgbBNwDgvYGrpQCAf+Xf1vkWiUSbN28eO3ZsSUmJn5+fr6/vmx+P0r6Zb+dtRLfw+fx169bRN3U6nVqtViqVCoWCw+GgyTz+/v5Hjx6trKy0t7dHr4bjOEmSUqkUja8rFAqSJFH7kiVL1q9fj2HYPiOBQPD555+PHTsWw7Bff/318ePHdnZ21kZWRihDfcWKFX/99RfahmfPnk2fPj0hIUEikfzL3Qkah0YffBsMBr1eXw9vhGZ46HS6WlM9mifCSK1Wm3pDGgTU58LeQJdiDQaDRqNpLNOb6hTqmrRabf30UQ0c7AQaSZLoLwU6DbQ3DAaDSqV6+/gbx/HWrVuzWKz3vg9xI6FQKBKJ7O3tKYrSarUYhpmbm48cOZL5rUVERJw6dYoeVtdqtXSui4uLC3qWi4tLWFiYTCbjcrlarRbH8QcPHpw5cwZVetFqtXRE8epnT09P//XXX2fMmFGnfzh8Ph/6apNo9NVO0FSMengj9CcnFAqhcAHaGwRBMJc9a87QYQPKOKDgW6vV8ng86NDRubpOp4PaL4her6coCqpFoXBTpVJxuVzoQtHe0Ol0fD6/0V03Y24wzkCSJAqpuVwu6glRoILjeFVVlUwmU/8vlUqlVqvlcnlxcfHq1atrnUj8/PPPkyZNqtPxPh6P1+h2e9PAaRpTMerhjVgsllqt5vF49Fzs5ozNZmu1WjhyIGq1Gs0HMvWGNAgGgwFGU2pFFRB8IyRJwp8J2g/QadBQqCoQCLCmgsVi1RqkowMVB6PXPis3N3fPnj30zeDg4NGjR3OM6n6TQX2DL/VtoUsEjf1CwfsC+wG8FvyZvAr2BgL7gQa74rWdRjMfgl25cqVKpTp+/DhJkpGRkZs2bYIy8E0YBN8AAAAAAKbk6Oh46NChJUuWKBSKsLAwSNBq2iD4BgAAAAAwMVSJRafTQeTd5EFeJgAAAABAQ6kkBolJTR6MfP+7KRTNPCmNhuM4zKijsdls2Bs02BU0FovFZrOh00BYLBaEFAiO49Bp0OBowgSdRjPR6EsNAgBAgwXTyAAAANQCp5sAAFBXIPIGAABQCwTfAAAAAAAA1BMIvgEAAAAAAKgnEHwDAAAAAABQTyD4BgAAAAAAoJ5AqUEsLy/vypUrNTU17dq1a9u2LYfz/+0TrVZ748aNFy9eODs79+rVy9bWFsMwhUJx/fr1tLQ0Ly+vXr16WVhYYBhWXV19+fLlvLy8wMDAqKgokUhkus8E3oPXfstMSUlJN27cYLPZUVFRgYGB6Ndy9+7dZ8+ecbncdu3atWnThsViqdXq+Pj4iooKHMdJkvT39w8NDTXRZwLvR1pa2vXr1/V6fYcOHdq0aVNrVuVrfzlpaWkvXrxA1aUEAkG3bt3EYnFJScnVq1dLS0tbtWrVuXNnWFmjsXvzUYCiqISEhLi4OLFY3L17d29v75qamtu3b2u1WjQ3F8dxZ2fnyMjIkpKS+Ph4g8GAas+1b9/e0dHRdB8L/Fd6vf7+/fuPHj2ytrbu3bu3s7Pzax92584dgUAQERHxhn7mzRELaCya+8h3cXHxjh07ampqnJ2dDx8+fPPmTea9FEWdOXPm/Pnz3t7eqampe/bsUalUFEUdOXLk9u3bfn5+CQkJhw4d0uv1Go1m//79SUlJvr6+V69ejY6OhhqOjRpBELW+ZXSApD1//nz37t0WFhZcLvfHH39MSUmhKOrkyZPHjx93d3c3MzPbu3fv3bt30W/s8OHDt27devDgwf3797Ozs033scB7kJOT8+OPP2IYZmlpuW/fvkePHr35l6PT6TAMu3jx4unTpx88eBAXF/f48WOtViuXy3ft2pWbm+vh4XH69Olz586Z7jOB90ClUr35KHD//v1Dhw7Z29ur1eoffvihqKhIo9E8evQoLi4uPj7+/v37+/fvv3fvHoZhCQkJR48eRe0PHjyorq423ccC7wH6Pbi7u5eUlOzcubO8vLzWAyiKunnz5q5du7Kyst7Qz5SUlLwhYgGNSHM/Z7p37x6Xy509ezaXyxWLxVevXu3YsaNQKET3lpSU3Lt3b+zYsZGRkZ07d167du2zZ8+cnZ2fP38+Z86cwMDAsLCw77//PiMjQ6/XZ2VlLV++3MnJydvbe+/evb169XJzczP15wPvKCcnp9a3nJ6eHhwcjO6lKOrKlSshISETJkzAMGznzp3Xrl1zdnbOysoaM2ZMly5dMAwzGAx3797t0qVLSUmJg4PDihUr0O8Kzsoau9jYWGdn52nTpqGVQa5cudKqVSsul/t3vxypVOrj41NeXj5lypSoqCi6+PeNGzcUCsX8+fOtrKwcHR1PnDjRuXNnOzs7U38+8I5SU1PfcBTQaDRXrlzp2rXrqFGjDAbD999/HxsbO27cuK+++gr9Hq5cuYJh2Mcff4xGN/v06TN58mT0XOg0GjWFQhEbGztkyJC+fftqNJpvvvkmPj5+4MCB9AO0Wu3vv//+5MkTgUDAZrP/rp8JCwu7f//+GyIW0Ig095Hv3Nxcb29vdOD08fFRKpUVFRX0vUVFRRRFeXl5YRiGDpA5OTl5eXlmZmbospGDg4NEIsk1sra2RgdOV1dXPp+fn59v0k8G/pNa37KFhUVubi59r0qlKioqatGiBbrp7++fl5fHZrOnTZsWGRmJDpZyuRxddC4sLORyuVevXj1y5EhSUhIUfm7UdDpdQUGBj48POiL6+/uXl5fLZLI3/HLy8/NlMplKpSotLf3tt98uXryoVCoxDMvKynJxcbG0tMQwzNvb22AwlJaWmvTDgf/kzUcBmUxWXV2NOg0Oh+Pt7Z2Xl2cwGNCKhiUlJRcuXBgwYICTk5NWqy0vL9fpdMePHz958mRJSQl0Go1aeXm5Vqv19vZGKWeenp61rn/q9Xo3N7eFCxcGBgYSBPF3/UxVVVVeXt4bIhbQiDTr4JsgCIVCYWZmhm4KBAIcx9VqNf0ApVLJ5XJRIiabzRYKhUqlsrq6WiAQMBvlcnlNTY1YLEZ/J1wul8/nKxQK030y8F/J5XI+n1/rq6fv1Wq1BoOBTugUiUR6vZ4kSXt7ez6fj2HYhQsXUlJS+vTpg4ZCMzIy8vPzKyoqtm/ffunSJdN9LPBfabVajUZjbm6ObopEIoIgNBrNm385JSUl+fn5ycnJKpXq8uXLW7duVSgUKpVKJBKhuIrH47HZbJVKZbpPBv4ruVz+hqOASqUiCILuNMRisUajQSlJFEWdP3/exsamU6dOGIZVVVXl5+enpKTIZLJnz56tXbs2JyfHdB8L/FcKhYLNZgsEAnTTzMxMqVSSJEk/wMzMrFevXh4eHijyfm0/Q5Kk0ugNEQtoRJp72gnqKBHqf9W6t9YlPxaLxWxB/2e+DrMdNFKvDjUxv1A0NarWzwDdJAgiJibm0qVLY8eODQgIwDBswIAB/fv3R/8/duzYxYsXO3bsSPeqoHFhGdX6667126j1FIPBgIa1goODRSJRfn7+unXrHjx4wOFw3vA6oNF581GAxWK9+ttALWVlZYmJiaNHj0YjmhKJZNq0ae7u7ig7fN26dVeuXPnkk0/q5UOAOv9h/N2fOR15/10/U6t7eTViAY1Isx75ZrPZ5ubmNTU16CY6g6RPK9H/DQYDmmlHEAQ66bS2ttYaoUaVSiUxUiqV9AUjjUYjkUhM98nAf2VpaanRaGp9y/S96NIHPRauVCr5fL5QKNRqtfv27btx48acOXO6du2KnmtnZ+fn54ce6evrq9VqYYCz8eLxeOhiF3NMi1nU4tVfDpqVGxgYiB5mb29vZ2dXVVVlYWGhVCrRsVOj0RAEwex8QKNjYWGhUCj+7iggEok4HA49Fo7S0tAVkqdPnwqFwpYtW6K7cBz39fW1t7fHMEwoFLq5uVVVVUGM1XiZmZmRJEkPUcvlcjMzs1dP1f6xnzE3ekPEAhqRZh18o1TLjIwMFAwlJyebm5tbW1tTFGUwGCiKcnJyYrPZaWlpaPJlcXGxt7e3u7u7UqlE1wFzc3NlMpmHh4enp2dFRUVBQQGGYZmZmVqtFmZbNmru7u4qlYr5Lbu7u6NwiiAIgUDg6ur64sULDMNIkkxKSvLw8OByuWfOnHn58uWcOXOCg4P1ej1BEAaDYe/evRcuXEAvm5KSIpFIXq1aCBoLLpfr5uaWmpqKEgaSkpLQxA+SJFFhuFd/OV5eXomJiVu2bEGH0sLCwtLSUnd3d29v7/z8/LKyMtT58Hg8BwcHU38+8O48PDyqqqpePQoYDAaSJC0tLW1sbFCnodFo0tLSPDw82Gw2RVGpqakeHh50pJ6bm/v9999nZmaiqCs7O9vFxQXSvhsvOzs7kUiUkpKCIu/MzEwfHx907ECdxtv0M/b29hKJxNPT89WIpd4/EHgPmnvaSfv27ePi4tavX+/k5PTkyZPx48cLBIIXL1788ccfn376qaura7du3X7//feUlJT09HQXF5fg4GA+nx8ZGfnzzz+HhYU9e/asVatWXl5eJEkGBgZu27YtODg4MTGxc+fOTk5Opv5w4N25urrW+pZ9fX0pijpw4ACXy500aVLv3r137ty5efNmiqLy8vIWLlyYm5v7559/6nS6bdu2obA7IiJi+vTp4eHh0dHROTk5er0+NTV18uTJMDm9UevZs+fGjRs3bdokFotTUlJmzpzJ4XDu379/8eLFhQsXvvrL8fHxsbS0PHfu3Lp16/z8/J4+fRoQENCqVSuCIGJjY7ds2eLt7f348eMPPvjAxsbG1B8OvDt/f/+AgADmUcDZ2bmysnLr1q0ffPBBREREv3799u7dKzNSKpXdu3dH45f5+fkRERF0eO3q6mpvb79169bw8HCpVIrjeO/evU394cC7EwqFffv2PXr0aGFhYX5+vrm5OZqXf/HixaSkpNmzZ4vFYvRInU5Hh+Ov9jNsNrtdu3b379+vFbGY9MOBd/T/5a02T6WlpXfv3q2pqQkLCwsJCcFxvKqqSiqVBgUFicVig8Hw8OHD5ORkR0fHDh06oNIEarU6ISFBKpW6u7t37NgRXU1WKBT37t3Lz8/38/OLjIxEE+9A4/Xqt4zGqHAcRyULMjIyHjx4gE7hvLy8ZDJZcnIyumaCRjXs7OwCAwMpikpKSnr69CmHw4mIiKBTUEDjlZeXd//+fa1WGxERgdZXKikpycvLCw0N5fF4r+0fysrK4uLiysrKfHx82rZtixqrqqru3btXWloaFBQUERFBVxkDjdSrRwGNRvP8+XNPT09UBeXFixePHz8WCASdOnVCJXHQObmdnR3zuodSqYyPj8/OznZ0dIyMjESLu4HGiyTJp0aWlpadO3dGX2hubm5lZWVwcDBaKIeiqLS0NLFY7Orq+nf9zGsjFpN+MvCOIPgGAAAAAACgnjT3nG8AAAAAAADqDQTfAAAAAAAA1BMIvgEAAAAAAKgnEHwDAAAAAABQTyD4BgAAAAAAoJ409zrfAADwfpWUlKSkpOTl5fXr18/W1lahUPzyyy9SqXTixImtW7c29dYBAAAwMQi+AQDgfaqurl63bl1KSkrXrl0rKip2796dmpp6586dli1bQvANAAAA6nwDAMB7dvLkyTVr1mzduvXWrVsdO3bs0aOHTqfjcrmwjA4AAADI+QYAgPfM0tKysrJy5syZLBarV69eLBZLIBBA5A0AAACCbwAAeP9IkuTz+WKx+MCBA/fu3TP15gAAAGhAIPgGAID3jCRJiUTy/fffW1tbz58/v7i42NRbBAAAoKGA4BsAAN4ziqJ0Op2/v/+ePXsKCwsXLlwol8tNvVEAAAAaBAi+AQDgPeNwOGZmZgRBBAcHd+jQ4ezZswsXLnz58qWptwsAAIDpQbUTAAB4zwiC0Gq1QqEQwzCNRmMwGAiCEIlEPB7P1JsGAADAxCD4BgAAAAAAAKsf/w9htiHyBZG7ewAAAABJRU5ErkJggg==)

first application, we can compute out-of-sample forecasts of average budget shares when the price changes. We focus on one of the categories, petrol. The motivation for this exercise is twofold. First, we want to showcase that our methodology can deliver informative bounds of quantities of interest. Second, fossil fuels prices are usually highly variable. This means that understanding the effects of such price changes on the problem of budget allocation in households is empirically relevant. This question may also be important for some potential users of this methodology, like a regulator trying to impose a green-tax. We study the following counterfactual question: What would be the average budget share of petrol in time T +1 , if the price of petrol p T +1 , pet is κ · 100 percent higher than p T, pet (i.e., p T +1 , pet = (1 + κ ) p T, pet )?

For simplicity, we consider κ that takes values in { 0 , 0 . 01 , · · · , 0 . 10 } (i.e., at most a 10 percent price increase). We set the (random) interest rate faced by the single-individual households r T +1 = 0 . 06 a . s . (roughly 1 percent increase over the average interest rate) and the support of the random discount factor to [0 . 975 , 1] . 67 The counterfactual moment is

<!-- formula-not-decoded -->

where θ pet ∈ [0 , 1] is the average budget share of petrol. Note that interest rate cancels out such that the budget share depends only on spot prices. For ED -rationalizability we do not need to specify the expenditure level at T + 1 , as the model endogenously predicts an expenditure level for a new price. This is because ED -rationalizability generalizes quasilinear rationalizabilty by adding discounting. However, adding discounting does not affect the model's budget-free nature (Gauthier, 2018).

The 95 percent bounds for the average counterfactual petrol share are depicted in Figure 1. Note that conditions of the Proposition 1 are satisfied. Hence, the sets of interest are connected, which means it is enough to depict the minimal and maximal shares that are not rejected by our test. 68 As expected, demand for petrol is decreasing in the price of petrol. The maximal and the minimal drops in shares associated with 10 percent price increase are 1 . 25 to 1 . 1 percentage points, respectively. 69

d

67 We computed our counterfactual sets with the support of the random discount factor equal to [0 . 1 , 1] and = 1 a . s . . The results are similar to those with [0 . 975 , 1] and available upon request.

68 We searched for average budget share θ pet in the grid { 0 . 00 , 0 . 005 , · · · , 1 } .

69 The empirical budget share at t = T computed from our (mismeasured) data set for petrol is 6 percent.

## G. Data Availability

The data sets and replication codes underlying this article are available in Zenodo, at https:// doi.org/10.5281/zenodo.4007866 .