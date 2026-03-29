## Published, Review of Economic Studies

https://doi.org/10.1093/restud/rdaf047

## REVISITING THE NON-PARAMETRIC ANALYSIS OF TIME-INCONSISTENT PREFERENCES

## FEDERICO ECHENIQUE AND GERELT TSERENJIGMID

Abstract. We revisit the recent revealed preference analysis of sophisticated quasi-hyperbolic consumers by Blow, Browning, and Crawford (2021) (BBC). We show that BBC's revealed preference test is too lax. There are non-rationalizable data that would pass their test. A basic problem with their test is that it requires finding a certain endogenous elasticity, without regard to the rationalizing utility. Their approach motivates a more stringent test, also based on first-order conditions, that would connect the endogenous elasticity and utility: We show that this test is also too lax. Aside from testing, we also discuss the possibility of recovering model parameters. We show that, even when discount factors are exactly identified, the approach followed in BBC allows for incorrect parameter values to lie in their identified set.

Echenique: Department of Economics, UC Berkeley, fede@econ.berkeley.edu Tserenjigmid: Department of Economics, UC Santa Cruz, gtserenj@ucsc.edu . We are grateful to the editors, the referees, and to Laura Blow for comments on an earlier draft. An earlier version of this paper, see arXiv:2305.14125 , did not focus only on a critique of Blow et al. (2021), but was mainly devoted to a test of sophisticated time-inconsistent models.

## Introduction

Blow, Browning, and Crawford (2021) (hereafter, BBC) present a revealedpreference characterization of quasi-hyperbolic discounting preferences in demand theory. A consumer chooses over time, according to intertemporal utility tradeoffs that change over time because of their quasi-hyperbolic preferences. Following Afriat (1967) (in the general utility-maximization framework), and Browning (1989) (for dynamically consistent exponential discounting), BBC use a first-order approach . That is, BBC equate consistency of data and model with the existence of a solution to a system of equations arising from first-order conditions for utility maximization . These first-order conditions are Euler equations, as derived in Harris and Laibson (2001), for example.

We argue that BBC's first-order approach is problematic and may lead to incorrect inferences. In the cases of Afriat and Browning, one can show that the first-order approach is correct. It leads to a test that is passed if and only if the model explains the data (Section 1.1). For the general model of utility maximization (Afriat), or the model of exponential discounting (Browning), data are consistent with the first-order approach if and only if there is a utility that satisfies the conditions in the model, and that rationalizes the data as optimal choices.

We show that there is a problem in using the first-order approach for a quasihyperbolic agent. The first-order approach is too permissive. Say that a dataset is FOCs rationalizable if the system of first-order conditions can be satisfied. A dataset is equilibrium rationalizable if there is a utility such that the observed consumption is an equilibrium outcome of the quasi-hyperbolic model. Our Theorem 1 shows that there are datasets that are FOCs rationalizable, but not equilibrium rationalizable. Our Theorems 1 and 4 show that the first-order approach leads to incorrect inference about crucial model parameters.

Theorems 1 and 4 matter beyond their theoretical implications. BBC carry out an empirical application in which they emphasize the explanatory power of the quasihyperbolic model: ' The quasi-hyperbolic model provides a significantly more successful account of behaviour than the alternatives considered .' The gap between equilibrium rationalizability and FOCs rationalizability may call into question the conclusions drawn from their empirical results.

We should emphasize that the revealed-preference problem for quasi-hyperbolic discounting is very difficult, and BBC make progress. There are problems with the first-order approach, but it is at least tractable.

## 1. Quasi-hyperbolic discounting and consumption.

We first outline the model of quasi-hyperbolic discounting consumer choice. We shall focus on a three-period model because it is the simplest case in which the assumption of hyperbolic discounting has any bite.

A consumer chooses quantities of a single good in periods t = 1 , 2 , 3. She has a wealth m , and faces prices p t for consumption in period t . These prices may be interpreted as encoding interest rates. Given prices and wealth, a consumption stream x = ( x 1 , x 2 , x 3 ) is affordable if p · x := ∑ 3 t =1 p t x t ≤ m . The standard exponentialdiscounting model assumes that preferences over a consumption stream x ∈ R 3 + are described by a pair ( u, δ ), with u : R + → R , and δ &gt; 0. The consumer evaluates a consumption stream x by u ( x 1 ) + δ u ( x 2 ) + δ 2 u ( x 3 ) .

Under quasi-hyperbolic discounting, the consumer's preferences are described by a tuple ( u, β, δ ), with u : R + → R , and β, δ &gt; 0. The consumer evaluates a consumption path x by u ( x 1 ) + β [ δu ( x 2 ) + δ 2 u ( x 3 )]. A quasi-hyperbolic consumer chooses consumption that results from an equilibrium between their period-1 preferences and their period-2 preferences. We phrase this as a game played between two agents. Agent 1 chooses consumption in period 1, x 1 . Agent 2 chooses consumption in period 2, and consequently in period 3 because consumption in period 3 is determined by the consumer's overall budget. So Agent 2 chooses ( x 2 , x 3 ).

The relevant equilibrium notion embodies a form of sequential rationality: it is a subgame-perfect Nash equilibrium. A subgame-perfect equilibrium can be described by backward induction: In period 2, given x 1 , Agent 2 maximizes u ( x 2 ) + βδu ( x 3 ) subject to x 2 , x 3 ≥ 0 and p 2 x 2 + p 3 x 3 ≤ m -p 1 x 1 . Let s ( x 1 ) = ( s 2 ( x 1 ) , s 3 ( x 1 )) denote a solution to Agent 2's problem, as a consumption vector in periods 2 and 3, and as a function of the period-1 choice x 1 .

Agent 1 then solves the problem of choosing period-1 consumption x 1 to maximize

<!-- formula-not-decoded -->

subject to x 1 ≥ 0 and p 1 x 1 ≤ m . If x ∗ 1 is an optimal choice for Agent 1, we say that the pair ( x ∗ 1 , s ) is a subgame-perfect Nash equilibrium of the game induced by ( u, β, δ ). In the sequel, we simply write equilibrium to refer to a subgame-perfect Nash equilibrium. An equilibrium outcome of the game defined by ( u, β, δ ), for fixed prices p and budget m , is then a consumption stream x = ( x 1 , x 2 , x 3 ) for which there exists an equilibrium ( x ∗ 1 , s ) with x ∗ 1 = x 1 and ( x 2 , x 3 ) = s ( x ∗ 1 ).

A dataset consists of a pair ( x, p ), where x ∈ R 3 ++ and p ∈ R 3 ++ . The interpretation is that we observe a consumption stream x , chosen when the prices are p , and income (or budget) is m := p · x . Importantly, x is the observed, or realized, consumption choice. Let U + be the set of all monotone increasing, C 2 , and strictly concave functions u : R + → R . We follow BBC in imposing strict concavity of utility, δ ≤ 1 and β &lt; 1. 1

We introduce the relevant notions of rationalizability: what it means for a dataset to be consistent with this particular theory of consumer choice.

Definition 1. A dataset ( x, p ) is equilibrium rationalizable by the sophisticated quasi-hyperbolic model if there exists ( u, β, δ ), with u ∈ U + , β ∈ (0 , 1), and δ ∈ (0 , 1], for which x is an equilibrium outcome of the game defined by ( u, β, δ ) for prices p and budget m = p · x . We say that ( u, δ, β ) is an equilibrium rationalization of ( x, p ). EQ denotes the set of equilibrium rationalizable datasets.

Equilibrium rationalizability requires checking complicated optimization and equilibrium properties. The literature on revealed preference theory, following the seminal work of Afriat (1967), often focuses on the data satisfying the first-order conditions of a model. We call this the first-order approach . BBC, while ostensibly about equilibria in the quasi-hyperbolic discounting model, actually formally uses the first-order approach. Our next definition is Definition 1 in BBC.

Definition 2. A dataset ( x, p ) is FOCs rationalizable by the sophisticated quasihyperbolic model if there exists ( u, β, δ, ( µ t ) 3 t =1 ) such that u ∈ U + , λ &gt; 0, β ∈ (0 , 1), δ ∈ (0 , 1], µ t ∈ (0 , 1) for t = 1 , 2, and µ 3 = 1 such that

<!-- formula-not-decoded -->

We say that the tuple ( u, β, δ, ( µ t ) 3 t =1 ) is a FOCs rationalization of ( x, p ). We also say ( u, β, δ ) is a FOCs rationalization of ( x, p ) if a desirable ( µ t ) 3 t =1 exists. Let FOC be the set of all datasets that are FOCs rationalizable.

FOCs rationalizability has a straightforward conceptual flaw: the elasticities µ t are not required to arise from the rationalizing utility u . BBC do not prove that

1 BBCexplicitly assume concavity, but implicitly strict concavity. They assume that the consumption function is differentiable, which rules out a zero of the second derivative of the instantaneous utility function (see Theorem C.3.2 in Mas-Colell (1985)). BBC also refer to Harris and Laibson (2001), who do assume strict concavity of utility.

FOC=EQ. In fact, as we show below, there is a data set ( x, p ) that is FOCs rationalizable but not equilibrium rationalizable.

For the first-order approach to be valid, two conditions are needed. First, any dataset that is FOCs rationalizable needs to be equilibrium rationalizable. Second, any FOCs rationalization of a dataset should also be an equilibrium rationalization. So model parameters that are recovered from the dataset are, in fact, consistent with the model. For general, and exponential, utility, the two conditions are satisfied (Section 1.1). For quasi-hyperbolic discounting, both conditions are violated.

Theorem 1. EQ ⊊ FOC. Moreover, there is a data set ( x, p ) ∈ EQ with a FOCs rationalization ( u, β, δ ) that is not an equilibrium rationalization of ( x, p ) .

The intuition behind Theorem 1, and our subsequent results, depends on the notion of absolute risk aversion. Let AR( x ) = -u ′′ ( x ) u ′ ( x ) be the coefficient of absolute risk aversion at x . Let R ( p ) = p 2 2 -p 1 p 3 p 3 ( p 1 -p 2 ) ; and PS be the set of data ( x, p ) with max( x 1 , x 3 ) ≤ x 2 , p 1 &gt; p 2 &gt; p 3 , and R ( p ) &gt; 1. In our proof, we show that any data in PS is FOCs rationalizable, but some of them are not equilibrium rationalizable. By leveraging the connection between µ t and u , we show that in order for ( x, p ) ∈ PS to be equilibrium rationalizable, AR( x 2 ) has to be at least R ( p ) times larger than AR( x 3 ), i.e., AR( x 2 ) AR ( x 3 ) ≥ R ( p ) &gt; 1.

We can make three observations from the inequality AR( x 2 ) AR( x 3 ) ≥ R ( p ) &gt; 1. First, any ( x, p ) ∈ PS with x 2 = x 3 is not equilibrium rationalizable because AR( x 2 ) = AR( x 3 ), which proves the first part of Theorem 1. Second, if ( x, p ) ∈ PS is equilibrium rationalizable, then a rationalizing utility function violates decreasing absolute risk aversion, which is assumed in most economic environments, and overwhelmingly supported by empirical evidence. Third, R ( p ) can be arbitrarily large. For example, R ( p ) &gt; k +( k +1) 2 ϵ when p = (1+( k +2) ϵ, 1+( k +1) ϵ, 1). Hence, for any rationalizing utility function u of ( x, p ), even allowing for increasing absolute risk aversion, the coefficient of absolute risk aversion needs to make arbitrarily large jumps (even when x 2 and x 3 are arbitrarily close). This excludes the class of utility functions assumed in Harris and Laibson (2001), where the coefficient of relative risk aversion is bounded (Theorem 5). This also rules out utility functions that are uniformly log-Lipschitz continuous (Theorem 5). Essentially because of the problematic properties of a FOCs rationalizing utility function, a data set ( x, p ) ∈ PS with max( x 1 , x 3 ) &lt; x 2 can be a local minimizer of Agent 1's objective function while it satisfies FOCs; which proves the second part of Theorem 1.

1.1. Exponential discounting model and the First-Order Approach. We argued that two conditions are needed for the first-order approach to be valid. To contrast with quasi-hyperbolic discounting, we consider exponentially discounted utility (EDU). A dataset ( x, p ) is EDU-rationalizable if it is equilibrium rationalizable by the quasi-hyperbolic model with some ( u, β, δ ) where β = 1. When this occurs, we say that ( u, δ ) is a EDU-rationalization of ( x, p ). Similarly, we say a dataset ( x, p ) is EDU-FOCs rationalizable if it is FOC rationalizable by the quasi-hyperbolic model with some ( u, β, δ ) where β = 1. EDU-FOCs is the test used by Browning (1989). The pair ( u, δ ) is an EDU-FOCs rationalization of ( x, p ).

Afriat (1967) shows that the first-order approach is valid for the general model of utility maximization. The same is true for exponential discounting:

Proposition 2. A dataset is EDU-FOCs rationalizable if and only if it is EDUrationalizable. Moreover, any EDU-FOCs rationalization ( u, δ ) of ( x, p ) is also an EDU-rationalization of ( x, p ) .

Proposition 2 is presented, without proof, to contrast with Theorem 1.

1.2. Quasi-hyperbolic model and Strong FOC. The disconnect between u and µ t in the definition of FOCs rationalization motivates our next definition. It suggests that u and µ t may be connected through Equation (2) (see Lemma 1 of BBC).

Definition 3. A dataset ( x, p ) is strongly FOCs rationalizable by the sophisticated quasi-hyperbolic model if there exists a FOCs rationalization ( u, β, δ, ( µ t ) 3 t =1 ) that satisfies

<!-- formula-not-decoded -->

Let Strong FOC be the set of all datasets that are strongly FOCs rationalizable.

We will derive Equation 2 from the utility maximization problems of Agent 1 and Agent 2 in Section 3. Under standard regularity conditions, the equation is necessary for equilibrium rationalization, i.e., EQ ⊆ Strong FOC.

̸

̸

Let D be the set of datasets ( x, p ) that satisfy x t = x s for all t = s , and I be the set of datasets with p 1 &gt; p 2 &gt; p 3 and p 1 /p 2 &gt; p 2 /p 3 .

## Theorem 3.

- (1) EQ ⊆ Strong FOC ⊊ FOC,
- (2) D ∩ FOC = D ∩ Strong FOC ⊊ Strong FOC,

## (3) and I ⊆ FOC.

To unpack the theorem, we discuss the different claims it contains. Statement (1) gives the obvious logical relations: EQ ⊆ Strong FOC ⊆ FOC; but in contrast with the message of Proposition 2 for exponential discounting, there is a gap between the notion of equilibrium and FOCs rationalization; the gap already appears in comparing FOCs and strong FOCs rationalizations. Strong FOC is a strict subset of FOC.

Statement (2) addresses the disconnect between u and µ t s in BBC. Theorem 3 says that, as long as consumption in different time periods is distinct, it is always possible to line up the µ t numbers with the intended rationalization. So strong FOCs seem to be too permissive as well. Further evidence on the permissiveness of strong FOCs is in Theorems 4 and 5.

Statement (3) provides additional evidence about the weakness of FOCs rationalization. No matter what the values of consumption are, as long as a dataset satisfies the assumption on prices in I, then it is FOCs rationalizable. It is worth mentioning that such prices are compatible with data that refute the exponential discounting model. 2

Our next result speaks to the use of FOCs rationalizability to recover the discount factors in a quasi-hyperbolic utility function. Discount factors matter critically for welfare comparisons and policy decisions, and estimating β and δ is part of the empirical exercise in BBC. But the proof of Proposition 1 in BBC shows that, whenever a dataset is FOCs rationalizable, it is without loss of generality to assume δ = 1. Our next result shows that this is problematic: there are datasets for which ( β, δ ) are point identified and δ &lt; 1, but δ = 1 is also in BBC's identified set.

## Theorem 4.

- (1) Let δ ∗ ∈ (0 , 1) and β ∗ ∈ (0 , 1) . There is ( x, p ) ∈ EQ such that: a) δ = δ ∗ and β = β ∗ for any equilibrium rationalization ( u, β, δ ) of ( x, p ) , and b) there is also a FOCs rationalization ( u, β ′ , δ ′ ) of ( x, p ) with δ ′ = 1 .
- (2) There are ( x, p ) ∈ D and ( u, β, δ ) such that ( u, β, δ ) is a Strong FOCs rationalization of ( x, p ) , but not an equilibrium rationalization of ( x, p ) .

Statement (1) of Theorem 4 means that δ &lt; 1 has additional empirical content when we focus on equilibrium rationalizability rather than FOCs. Statement (2) speaks to the possibility of using a FOCs, or Strong FOCs, rationalization in order

2 For example, by Theorem 1 of Echenique et al. (2020), the data set ( x, p ) ∈ I with x = (3 , 2 , 4) and p = (8 , 2 , 1) is not EDU-rationalizable.

to recover utility parameters. The theorem says that a rationalization may not have an equilibrium outcome that coincides with the data, which would mean that the rationalizing parameters could not generate the observed data. Theorem 4 challenges the analysis in Section 3.4 of BBC, in which they recover consumers' preferences based on a FOCs rationalization. The recovered preferences may not explain the data according to the quasi-hyperbolic model.

## 2. Robustness

Some of our results take essentially the form of counterexamples, or of families of counterexamples. Here we offer evidence that these are not, in some sense, 'knifedge.' We present first two classes of utility functions.

Given α ≥ 0, a function f : A ⊆ R → R ++ is α -logarithmically Lipschitz if, for all x ∈ A and t &gt; 0 so that x + t ∈ A , f ( x + t ) /f ( x ) ≤ (1 + t ) α . Observe that any non-increasing function is trivially α -logarithmically Lipschitz. Let U α be the set of all smooth, strictly monotone increasing, and strictly concave utility functions for which the coefficient of absolute risk aversion is α -logarithmically Lipschitz. Note that U α ⊆ U β for any β ≥ α ≥ 0 . The class U 0 contains all functions satisfying non-increasing absolute risk aversion. There is, of course, overwhelming empirical support for the assumption of non-increasing absolute risk aversion. 3 Let FOC α and Strong FOC α be the sets of data sets that are FOC rationalizable and strong FOC rationalizable by a utility function u ∈ U α .

We also consider the class of utility functions U HL ⊆ U that are assumed by Harris and Laibson (2001). For reasons of space, we do not include the complete definition of U HL : see assumptions U1-U4 on page 940 of their paper. The critical assumption for us is U4 which assumes that the coefficient of relative risk aversion is bounded away from 0 and + ∞ . The results of BBC rely on Harris and Laibson's results (for example, Lemma 1 in BBC is one of Harris and Laibson's results).

Theorem 5. Let α ≥ 0 . There is an open subset D o of D with the property that, for any ( x, p ) ∈ D o :

- (1) ( x, p ) is in FOC α but not equilibrium rationalizable with any utility u ∈ U α ∪ U HL ;
- (2) ( x, p ) is not in Strong FOC α .

3 See, for example, Cohn et al. (1975), Levy (1994), Guiso and Paiella (2008), Chiappori and Paiella (2011), and Paravisini et al. (2017).

Remark 1 . There is, as said, overwhelming empirical support for U 0 . Theorem 5 only requires U α , for some α ≥ 0. It is easy to prove even stronger versions of Theorem 5: We may require that there is a bounded function h : [0 , 1] → R + such that, for all u , there is ε ∈ (0 , 1) with AR( x + t ) / AR( x ) ≤ h ( t ) for all t ∈ (0 , ε ). The conclusion of the theorem holds in this case.

Remark 2 . The proof of Theorem 5 shows a stronger result: for any α ≥ 0, there is an open subset D o of D with the property that D o ⊆ FOC 0 and D o ∩ Strong FOC α = ∅ . Since FOC 0 ⊆ FOC α for any α ≥ 0, we obtain D o ⊆ FOC α . In fact, as we show in the proof, every ( x, p ) ∈ D o will be FOC rationalizable with a CRRA utility function.

Remark 3 . For additional robustness, we present a result like Theorem 5 for arbitrary compact sets of utilities in the Online Appendix A.1.

Remark 4 . The focus so far has been on a three-period model, which raises the possibility that the equivalence between EC and FOCS is valid for longer time horizons. It is, of course, easy to recreate the counterexamples in our proofs so that they occur in the last three periods of a problem with an arbitrary finite horizon. But we present a counterexample in Online Appendix A.2 in which the incompatibility arises at intermediate periods: not at the beginning nor at the end of the time horizon.

## 3. Proofs

As background for the proofs of Theorems 1-5, we derive convenient expressions for the model's first conditions. We derive the first-order conditions (FOCs) by backward induction. Agent 2 maximizes u ( x 2 )+ β δ u ( x 3 ) subject to the budget constraint. The FOC is

<!-- formula-not-decoded -->

Hence, x 3 = g ( x 2 ) where g := u ′-1 ( Au ′ ) and A = p 3 β δ p 2 . Note that g is continuous and strictly increasing. Agent 1 maximizes u ( x 1 ) + βδu ( x 2 ) + β δ 2 u ( x 3 ) subject to the budget constraint. Let

<!-- formula-not-decoded -->

Note that f is continuous and strictly decreasing. Agent 1 maximizes u ( f ( x 2 ) ) + β δ u ( x 2 ) + β δ 2 u ( g ( x 2 ) ) . The FOC gives

<!-- formula-not-decoded -->

since f ′ ( x 2 ) = -p 2 + p 3 g ′ ( x 2 ) p 1 . Then u ′ ( x 3 ) = p 3 β δ p 2 u ′ ( x 2 ), implies that

<!-- formula-not-decoded -->

Note that strong FOCs rationalizability is equivalent to ( u, β, δ ) satisfying Equations (3) and (4).

Let us see how Equation (1) is related to Equations (3) and (4). Define λ = δ u ′ ( x 1 )(1 -(1 -β ) µ 1 ) p 1 . We then obtain Equation (1) for t = 1 for any ( u, β, δ ) and µ 1 ∈ (0 , 1). Since µ 3 = 1, the FOC rationalizability is equivalent to

<!-- formula-not-decoded -->

Since u ′ ( x 2 ) = p 2 p 3 β δ u ′ ( g ( x 2 )) and u ′′ &lt; 0, the Implicit Function Theorem implies that g ′ ( x 2 ) = p 3 u ′′ ( x 2 ) p 2 βδu ′′ ( x 3 ) . Hence,

<!-- formula-not-decoded -->

Hence, Equations (4) and (5) are satisfied iff µ 2 = p 2 p 2 + p 3 g ′ ( x 2 ) iff Equation 2 is satisfied.

3.1. Proof of Theorem 1. Recall the following notation: i) AR( x ) = -u ′′ ( x ) u ′ ( x ) is the coefficient of absolute risk aversion at x ; ii) R ( p ) = p 2 2 -p 1 p 3 p 3 ( p 1 -p 2 ) ; and iii) PS is the set of data sets ( x, p ) with x 1 ≤ x 2 ≥ x 3 , p 1 &gt; p 2 &gt; p 3 , and R ( p ) &gt; 1. Let GPS be the set of data sets ( x, p ) with x 1 ≤ x 2 ≥ x 3 , p 2 p 3 &gt; p 1 p 2 &gt; 1. Note that PS ⊊ GPS.

Lemma 6. For any ( x, p ) ∈ Strong FOC, g ′ ( x 2 ) ≥ R ( p ) .

Proof. By Equation (3), we obtain p 3 δp 2 u ′ ( x 2 ) u ′ ( x 3 ) = β . By Equation (4), the fact g ′ ( x 2 ) = AR ( x 2 ) /AR ( x 3 ) &gt; 0, and the assumption x 1 ≤ x 2 (and hence 1 ≤ u ′ ( x 1 ) u ′ ( x 2 ) ),

<!-- formula-not-decoded -->

This implies that

Since u ′ ( x 2 ) ≤ u ′ ( x 3 ),

<!-- formula-not-decoded -->

Hence, we obtain the desired inequality. □

Lemma 7. GPS ⊆ FOC.

<!-- formula-not-decoded -->

Proof. Let ( x, p ) ∈ GPS . Let δ 1 , δ 2 ∈ [0 , p 1 p 2 -1) be such that i) δ 1 = 0 if x 1 = x 2 ; ii) δ 2 = 0 if x 2 = x 3 ; and iii) x 1 ≥ x 3 iff δ 1 ≤ δ 2 . Choose any u ∈ U + with u ′ ( x 1 ) = u ′ ( x 2 )(1 + δ 1 ) and u ′ ( x 3 ) = u ′ ( x 2 )(1 + δ 2 ).

Let δ = 1. To obtain Equation (5) for FOCs, we shall find β, µ 2 ∈ (0 , 1) such that

<!-- formula-not-decoded -->

From the second equality, we find β = p 3 p 2 (1+ δ 2 ) &lt; 1. Then from the first equality, we find

<!-- formula-not-decoded -->

Note that µ 2 &gt; 0 because δ 1 &lt; p 1 p 2 -1. Moreover, µ 2 &lt; 1 since p 3 p 2 (1+ δ 2 ) ≤ p 3 p 2 &lt; p 2 p 1 ≤ p 2 p 1 (1 + δ 1 ). Hence, we obtain desired β and µ 2 . Thus, ( x, p ) ∈ FOC. □

We can now wrap up the proof of Theorem 1. Let ( x, p ) ∈ PS with x 2 = x 3 . Since PS ⊆ GPS, Lemma 7 implies that ( x, p ) ∈ FOC . However, if ( x, p ) ∈ EQ ⊆ Strong FOC, then by Lemma 6 we should have g ′ ( x 2 ) ≥ R ( p ) &gt; 1. However, by Equation (6), x 2 = x 3 implies that g ′ ( x 2 ) = 1. Therefore, ( x, p ) ̸∈ EQ .

The second statement of Theorem 1 follows from the first part of Theorem 4, which proves that there are ( x, p ) ∈ EQ and ( u, β, δ ) such that ( u, β, δ ) is a FOCs rationalization of ( x, p ) but not an EQ rationalization of ( x, p ).

- 3.2. Proof of Theorem 3. It is obvious that EQ ⊆ Strong FOC ⊆ FOC. We proceed to show the other statements in the theorem.

Part 1: There exists a dataset in FOC that is not in Strong FOC. By Lemma 7, PS ⊊ FOC. By Lemma 6, if ( x, p ) ∈ Strong FOC, then g ′ ( x 2 ) ≥ R ( p ) &gt; 1. However, by Equation (6), if x 2 = x 3 , then we obtain g ′ ( x 2 ) = 1, a contradiction. Hence, for any ( x, p ) ∈ PS with x 2 = x 3 , ( x, p ) ∈ FOC but ( x, p ) ̸∈ Strong FOC.

̸

Part 2. Any dataset in FOC that has x t = x s for all t = s is strong FOCs rationalizable. We claim that, if (ˆ u, β, δ, ( µ t ) 3 t =1 ) is a FOCs rationalization, then we may find a strong FOCs rationalization ( u, β, δ, ( µ t ) 3 t =1 ) for which u ′ ( x t ) = ˆ u ′ ( x t ) for all t . To this end, let a t = ˆ u ′ ( x t ) &gt; 0, and choose b t &lt; 0 so that µ 2 = p 2 p 2 + p 3 G holds where G = a 3 b 2 a 2 b 3 . Note that ( u, β, δ, ( µ t ) 3 t =1 ) will be a FOCs rationalization if u ′ ( x t ) = a t . Moreover, Equation (2) will be satisfied if u ′′ ( x t ) = b t .

̸

̸

̸

Consider the function h t ( x ) = a t + b t ( x -x t ). Note that h t is monotone decreasing and that a t = h t ( x t ) &lt; h s ( x s ) = a s when x s &lt; x t , as ˆ u is strictly concave. Given that x s = x t for t = s we may find disjoint neighborhoods N t of each x t so that h t is

smaller on N t than h s on N s when x s &lt; x t , and greater on N t than h s on N s when x s &gt; x t . Define a function h : R + → R by letting h equal h t on N t , h (0) &gt; sup { h t ( x ) : x ∈ N t , 1 ≤ t ≤ 3 } , and by linear interpolation on R + \ ( { 0 } ∪ ( ∪ t N t )). Then h is monotone decreasing, h ( x t ) = ˆ u ′ ( x t ), and h ′ t = b t for all t . Letting u ( x ) = ∫ x 0 h ( z ) d z , we have u ′ ( x t ) = h ( x t ) = a t and u ′′ ( x t ) = h ′ ( x t ) = b t . Finally, by choosing data in Strong FOC that is not in D we obtain the strict inclusion.

Part 3. I ⊆ FOC follows from Proposition 1 of BBC.

3.3. Proof of Theorem 4. Part 1: To prove the first statement in the Theorem, fix δ ∗ , β ∗ ∈ (0 , 1), and consider data ( x, p ) with x 1 = x 2 = x 3 and p 3 = 1 and

<!-- formula-not-decoded -->

Let ( u, β, δ ) be an arbitrary equilibrium rationalization of the data. We claim that δ = δ ∗ and β = β ∗ .

Since x 2 = x 3 , by Equation (3), we have 1 = βδp 2 ; i.e., βδ = β ∗ δ ∗ . Moreover, we also obtain A = 1, which means that g ′ ( x 2 ) = 1. Since x 1 = x 2 , by Equation (4), we have

<!-- formula-not-decoded -->

Hence, δ = δ ∗ and β = β ∗ .

Note that p 2 p 3 &gt; p 1 p 2 &gt; 1. Hence, ( x, p ) ∈ GPS. By Lemma 7, ( x, p ) ∈ FOC .

On the other hand, the proof of Proposition 1 in BBC shows that, whenever a dataset is FOCs rationalizable, then it is without loss of generality to set δ = 1. It is in fact easy to show that the data is FOCs rationalizable with ( u, β ′ , δ ′ ) with δ ′ = 1 and β ′ = β ∗ δ ∗ by setting µ 2 = 1 -β ∗ ( δ ∗ ) 2 1 -( β ∗ ) 2 ( δ ∗ ) 2 .

Part 2: Finally, we prove the second statement in Theorem 4. Consider a dataset with x 1 = 0 . 04 , x 2 = 0 . 05, x 3 = 0 . 4698, and prices p 1 = 3 . 0969, p 2 = 2, p 3 = 1 (consequently, m = 0 . 694). We claim that ( u, β, δ ), with β = δ = 0 . 8, and u ( x ) = x -x 3 3 when x ∈ (0 , 1), is a strong FOCs rationalization. Indeed, note that g ( x ) = √ 1 -A (1 -x 2 ) where A = 0 . 78125. By direct calculation, we obtain

<!-- formula-not-decoded -->

To verify strong FOCs rationalizability, note that Equation (3) is satisfied since

<!-- formula-not-decoded -->

Moreover, Equation (4) is satisfied since

<!-- formula-not-decoded -->

To check the equilibrium rationalizability, we consider the second-order condition for Agent 1. Recall from the discussion at the start of Section 3 that Agent 1's objective function is

<!-- formula-not-decoded -->

Hence, the SOC is

<!-- formula-not-decoded -->

Using Equations (3) and (4), we can further simplify and obtain

<!-- formula-not-decoded -->

However, we have

<!-- formula-not-decoded -->

so (7) is violated. Hence, the bundle x is a local minimizer for Agent 1's problem.

3.4. Proof of Theorem 5. Note that there is 0 &lt; σ &lt; ¯ σ such that any u in U HL satisfies that xu ′′ ( x ) /u ′ ( x ) ∈ [ -¯ σ, -σ ] (this is property U4 in Harris and Laibson (2001)). Take any k &gt; max(2 α -1 , ¯ σ/σ ) and let p k = (2 + 1 k +1 , 2 , 1). The function R ( p ) = p 2 2 -p 1 p 3 p 3 ( p 1 -p 2 ) evaluates to 2 k +1 at p = p k and is continuous in a neighborhood of p k . Let P ⊊ R 3 ++ be a neighborhood of p k with the property that R ( p ) &gt; 2 k for any p ∈ P . Let D o be the set of all data sets ( x, p ) with p ∈ P , x 1 ∈ (0 , 0 . 5) , x 3 ∈ (0 . 5 , 1) and x 2 ∈ (1 , 1 . 5). Note that D o ⊆ GPS and D o ⊆ D.

Suppose ( u, β, δ ) is an equilibrium rationalization of ( x, p ) ∈ D o with u ∈ U α ∪ U HL . Then by Lemma 6, g ′ ( x 2 ) = AR ( x 2 ) AR ( x 3 ) ≥ R ( p ) &gt; 2 k . However, if u ∈ U α , then AR ( x 2 ) AR ( x 3 ) ≤ (1 + x 2 -x 3 ) α ≤ 2 α &lt; 2 k, a contradiction. Suppose now u ∈ U HL . Denote by RR( x ) = -xu ′′ ( x ) /u ′ ( x ) the coefficient of relative risk aversion of u . Then we obtain

<!-- formula-not-decoded -->

a contradiction. Hence, we conclude that ( u, β, δ ) is not an equilibrium rationalization of ( x, p ). The above also show that ( x, p ) is not in Strong FOC α .

We shall prove that D o ⊆ FOC α . Since FOC 0 ⊆ FOC α , it is enough to prove that D o ⊆ FOC 0 . Consider ( u, β, δ, ( µ t ) 3 t =1 ) such that u ( x ) = x γ , δ = 1, β = p 3 p 2 ( x 3 x 2 ) 1 -γ , and

<!-- formula-not-decoded -->

Note that u ∈ U 0 when γ ∈ (0 , 1). We shall show that every ( x, p ) ∈ D o is FOCs rationalizable with ( u, β, δ, ( µ t ) 3 t =1 ) for some γ ∈ (0 , 1). Note that with the above specifications, we obtain the FOCs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Hence, it is sufficient to prove that there is γ ∈ (0 , 1) such that β ∈ (0 , 1) and µ 2 ∈ (0 , 1). Recall that since D o ⊆ GPS , we have x 2 &gt; max( x 1 , x 3 ) and p 2 p 3 &gt; p 1 p 2 &gt; 1. Since x 3 x 2 &lt; 1 and p 3 p 2 &lt; 1, we have β &lt; 1 for any γ ∈ (0 , 1). In order to have µ 2 &lt; 1, we need p 2 p 1 ( x 2 x 1 ) 1 -γ &gt; β = p 3 p 2 ( x 3 x 2 ) 1 -γ ; equivalently, p 2 2 p 1 p 3 &gt; ( x 1 x 3 x 2 2 ) 1 -γ . The last inequality holds for any γ ∈ (0 , 1) since p 2 2 p 1 p 3 &gt; 1 and 1 &gt; x 1 x 3 x 2 2 . Lastly, to have µ 2 &gt; 0, we need p 2 p 1 ( x 2 x 1 ) 1 -γ &lt; 1, which holds when γ is close to 1.

## References

- Afriat, S. N. (1967): 'The construction of utility functions from expenditure data,' International economic review , 8, 67-77.
- Aliprantis, C. D. and K. C. Border (2006): Infinite Dimensional Analysis: A Hitchhiker's Guide , Springer.
- Blow, L., M. Browning, and I. Crawford (2021): 'Non-parametric Analysis of TimeInconsistent Preferences,' The Review of Economic Studies , 88, 2687-2734.
- Browning, M. (1989): 'A nonparametric test of the life-cycle rational expections hypothesis,' International Economic Review , 979-992.
- Chiappori, P.-A. and M. Paiella (2011): 'Relative Risk Aversion is Constant: Evidence from Panel Data,' Journal of the European Economic Association , 9, 1021-1052.
- Cohn, R. A., W. G. Lewellen, R. C. Lease, and G. G. Schlarbaum (1975): 'Individual Investor Risk Aversion and Investment Portfolio Composition,' The Journal of Finance , 30, 605620.
- Echenique, F., T. Imai, and K. Saito (2020): 'Testable implications of models of intertemporal choice: Exponential discounting and its generalizations,' American Economic Journal: Microeconomics , 12, 114-43.

- Guiso, L. and M. Paiella (2008): 'Risk Aversion, Wealth, and Background Risk,' Journal of the European Economic Association , 6, 1109-1150.
- Harris, C. and D. Laibson (2001): 'Dynamic choices of hyperbolic consumers,' Econometrica , 69, 935-957.
- Hirsch, M. W. (2012): Differential topology , vol. 33, Springer Science &amp; Business Media.
- Levy, H. (1994): 'Absolute and relative risk aversion: An experimental study,' Journal of Risk and uncertainty , 8, 289-307.
- Mas-Colell, A. (1985): The theory of general economic equilibrium: A differentiable approach , 9, Cambridge University Press.
- Paravisini, D., V. Rappoport, and E. Ravina (2017): 'Risk Aversion and Wealth: Evidence from Person-to-Person Lending Portfolios,' Management Science , 63, 279-297.

## Online Appendix

Revisiting the Non-Parametric Analysis of Time-Inconsistent Preferences

Federico Echenique and Gerelt Tserenjigmid

## Appendix A. Complementary results

A.1. An additional robustness result: Using a standard analytic argument, in the following, we prove a result like Theorem 5 for arbitrary compact sets of utilities. Let C r ([0 , K ]) denote the set of functions u : [0 , K ] → R that have continuous derivatives up to r , with 2 ≤ r ≤ ∞ . Endow this space with a Hausdorff topology. For example, under the weak topology, C r is a Banach space when r &lt; ∞ , and a separable, complete, and locally convex topological vector space when r = ∞ (see Hirsch (2012)). Let U r + ⊆ C r ([0 , K ]) be the collection of all strictly monotonically increasing and strictly convex functions in C r ([0 , K ]). The real number K is large and can be chosen without loss of generality (large enough to exceed any of the consumption values that may be relevant for the family of budgets in our set of data).

Theorem 8. For any compact ¯ U ⊆ U r + , there is an open subset D o of D for which D o ⊆ FOC but no data in D o is equilibrium rationalizable with any utility u ∈ ¯ U .

Proof. Let m 2 ( x 1 , p, I, u, β, δ ) be the solution to the problem of maximizing u ( x 2 ) + β δ u ( x 3 ) subject to p 2 x 2 + p 3 x 3 ≤ I -p 1 x 1 . By the strict concavity of u , the maximization problem has a unique solution. By Berge's maximum theorem (see Aliprantis and Border (2006)), m 2 is a continuous function. Let m 1 be the set of solutions to the problem of maximizing u ( x 1 ) + v ( m 2 ( x 1 , p, I, u, β, δ )) subject to p 1 x 1 ≤ I where v ( x 2 , x 3 ) = β δ u ( x 2 ) + β δ 2 u ( x 3 ). The set of solutions is non-empty, and the correspondence m 1 ( p, I, u, β, δ ) is upper hemicontinuous (again by the maximum theorem). A dataset ( x, p ) is rationalizable by u ∈ U r + and ( β, δ ) ∈ (0 , 1) × (0 , 1] iff x 1 ∈ m 1 ( p, I, u ) and ( x 1 , x 2 ) = m 2 ( x 1 , p, I, u ) where I = p 1 x 1 + p 2 x 2 + p 3 x 3 .

Now let D o be an open subset of GPS that contains the data ( x, p ) in the proof of Theorem 1. Then D o ⊆ FOC by Lemma 7. We claim that there is an open neighborhood of ( x, p ) in D o that is disjoint from EQ. Suppose, towards a contradiction, that there are datasets arbitrarily close to ( x, p ) that are in EQ. Choose a sequence ( x k , p k ) of such data with ( x, p ) = lim k →∞ ( x k , p k ). For each k there is a utility u k ∈ U and β k , δ k such that ( x k , p k ) is equilibrium rationalizable by ( u k , β k , δ k ).

By compactness of U , there is a subnet of ( u k , β k , δ k ) that converges to ( u, β, δ ) ∈ U × [0 , 1] × [0 , 1]. By continuity of m 2 and upper hemicontinuity of m 1 , ( x, p ) is rationalized by ( u, β, δ ).

Note that if β or δ are zero, then the rationalization would require x 2 or x 3 to be zero. So we must have β, δ &gt; 0. Note also that we have already ruled out β = 1 (Lemma 7)

in our rationalizing example. We are then left with a putative rationalization by a strictly concave u . A contradiction to Theorem 1. □

A.2. A five period example. Our paper focuses on a three-period model, which raises the possibility that the equivalence between EC and FOCS is valid for longer time horizons. It is, of course, easy to recreate the counterexamples in our proofs so that they occur in the last three periods of a problem with an arbitrary finite horizon. In this section, we present a counterexample in which the incompatibility arises at intermediate periods: not the beginning nor the end of the time horizon.

We present a class of examples with five periods. We label time t = 0 , 1 , 2 , 3 , 4. The problem arises, as before, with the consumption in periods 1 , 2 , 3 but these are now 'interior' periods: neither initial nor terminal.

A data set is a pair ( x, p ) = (( x 1 , x 2 , x 3 , ) , ( p 1 , p 2 , p 3 )) and p 0 , p 4 , x 0 , x 4 that are unobservable to an analyst. 4 We say ( x, p ) is equilibrium rationalizable if we can find p 0 , p 4 , x 0 , x 4 such that (˜ x = ( x 0 , x 1 , x 2 , x 3 , x 4 ) , ˜ p = ( p 0 , p 1 , p 2 , p 3 , p 4 )) is equilibrium rationalizable in the sense of Definition 1.

Recall that GPS be the set of data sets ( x, p ) with x 1 ≤ x 2 ≥ x 3 and p 2 p 3 &gt; p 1 p 2 &gt; 1. Let GPS ∗ be the subset of GPS such that x 2 = x 3 and R ( p )( p 3 p 2 -1 9 ) &gt; 1 (recall that R ( p ) = p 2 2 -p 1 p 3 p 3 ( p 1 -p 2 ) ). For example, when p is in a neighborhood of ˆ p = (4 , 3 , 1), the inequality R ( p )( p 3 p 2 -1 9 ) &gt; 1 is satisfied.

We will prove that any ( x, p ) ∈ GPS ∗ is FOCs rationalizable but not equilibrium rationalizable.

Let us fix ( β, δ, u ) ∈ (0 , 1) × (0 , 1] ×U + and derive the first-order conditions.

Agent 3 maximizes u ( x 3 )+ β δ u ( x 4 ) subject to the budget constraint. The FOC gives

<!-- formula-not-decoded -->

Let x 4 = h ( x 3 ) where h = u ′-1 ( Au ′ ) and A = p 4 β δ p 3 . Agent 2 maximizes u ( x 2 ) + βδu ( x 3 ) + β δ 2 u ( x 4 ) subject to the budget constraint and the fact that x 4 = h ( x 3 ). Let

<!-- formula-not-decoded -->

4 If p 0 , p 4 , x 0 , and x 4 are partially or fully observed, we can obtain stronger conclusions (when they are unobserved, it makes it easier for the data to be rationalizable).

Note that h is continuous, strictly increasing and σ is continuous, strictly decreasing. We may then take Agent 2 to choose x 3 to maximize

<!-- formula-not-decoded -->

The FOC gives

<!-- formula-not-decoded -->

since σ ′ ( x 3 ) = -p 3 + p 4 h ′ ( x 3 ) p 2 . Then u ′ ( x 4 ) = p 4 β δ p 3 u ′ ( x 3 ), implies that

<!-- formula-not-decoded -->

Let

<!-- formula-not-decoded -->

Let

<!-- formula-not-decoded -->

Then Agent 1 maximizes

<!-- formula-not-decoded -->

The FOC gives

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where the third equality is by Equation (9). Thus,

<!-- formula-not-decoded -->

For the above first-order condition to be valid, we need to have g ′ ≥ 0. If g ′ &lt; 0 at some x 2 , then for any β &lt; 1, δ, p 3 , and p 4 , there are some values of p 2 at which p 2 + g ′ ( x 2 ) ( p 3 + p 4 h ′ ( x 3 ) ) &gt; 0 &gt; β p 2 + g ′ ( x 2 ) ( p 3 + p 4 h ′ ( x 3 ) ) , so that the first-order condition cannot be satisfied. Hence, g ′ ≥ 0.

Let U ∗ + be the set of all utility functions in U + such that g is increasing for any ( β, δ, p ). If the utility function does not belong to this set, then using the first-order approach is not valid for reasons we have explained above. However, even if u ∈ U ∗ + , we still obtain an counter example.

Proposition 9. No dataset ( x, p ) ∈ GPS ∗ is equilibrium rationalizable by ( u, β, δ ) ∈ U ∗ + × (0 , 1] × (0 , 1] .

Proof of Proposition 9. Suppose, towards a contradiction, that ( x, p ) ∈ GPS ∗ is equilibrium rationalizable by ( u, β, δ ) ∈ U ∗ + × (0 , 1] × (0 , 1].

Step 1. β &lt; 1.

Since x 2 = x 3 , by Equation (9), we obtain

<!-- formula-not-decoded -->

which implies that and hence

<!-- formula-not-decoded -->

If β = 1 then we must have δp 2 = p 3 . Thus p 1 /p 2 &lt; p 2 /p 3 and Equation (10) gives

<!-- formula-not-decoded -->

and therefore x 1 &gt; x 2 . This is not possible given our assumption that x 1 ≤ x 2 .

Step 2. g ′ ( x 2 ) ≥ R ( p ).

By Step 1, we know that β &lt; 1. Hence Equation (12) implies that δp 2 &gt; p 3 , and therefore that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Equation (10), and the assumption that 1 ≤ u ′ ( x 1 ) u ′ ( x 2 ) ,

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

Since (in Step 1) β &lt; 1, we must have p 2 δ p 1 -1 &lt; 0. Thus,

<!-- formula-not-decoded -->

Hence, using Equation (13), we obtain that

<!-- formula-not-decoded -->

Again, by Equation (11), we have

Hence,

<!-- formula-not-decoded -->

Then by Equation (14),

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Step 3. If u ∈ U ∗ + , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given the putative rationalization ( u, β, δ ) the function h only depends on u and B = A -1 = β δ p 3 p 4 . So fix B &gt; 0. We shall prove that

<!-- formula-not-decoded -->

When h ′′ ≤ 0, the above inequality is trivially satisfied. So we may suppose without loss of generality that h ′′ &gt; 0.

Note that h is independent of ˜ β, ˜ δ, ˜ p for fixed B . Note that g is increasing iff

<!-- formula-not-decoded -->

is increasing iff (1 -(1 -˜ β )˜ p 3 ˜ p 3 +˜ p 4 h ′ ) u ′ is decreasing, since u ′-1 is strictly decreasing. Hence, g ′ ≥ 0 iff

<!-- formula-not-decoded -->

equivalently,

<!-- formula-not-decoded -->

Given that B = ˜ β ˜ δ ˜ p 3 ˜ p 4 ,

<!-- formula-not-decoded -->

which is independent of ˜ β . For any given B , we can find ˜ β &lt; 1 such that ˜ δ = 1 and ˜ p 3 ˜ p 4 is sufficiently large. Since

<!-- formula-not-decoded -->

when ˜ δ = 1, for any fixed B , we can find ˜ β &lt; 1 and ˜ p 3 ˜ p 4 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Equation (9), g satisfies

<!-- formula-not-decoded -->

By the chain rule,

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Suppose first that h ′′ ( x 3 ) ≤ 0. Then ( x, p ) ∈ GPS and Step 2 imply

<!-- formula-not-decoded -->

Suppose now h ′′ ( x 3 ) &gt; 0. Since u ′′ ( x 2 ) &lt; 0 and 1 0 . 9 ( β δ p 3 p 4 + h ′ ) ≥ u ′ -u ′′ h ′′ by Step 3, we have

<!-- formula-not-decoded -->

By Equation (13),

<!-- formula-not-decoded -->

Since βδ p 3 + p 4 h ′ ( x 3 ) β p 3 + p 4 h ′ ( x 3 ) ≤ 1, we have

<!-- formula-not-decoded -->

which leads to the desired inequality.

Note that ( x, p ) ∈ GPS ∗ means that x 2 = x 3 and R ( p )( p 3 p 2 -1 9 ) &gt; 1. Hence, by Step 4, we obtain 1 = AR ( x 2 ) AR ( x 3 ) ≥ R ( p )( p 3 p 2 -1 9 ) &gt; 1, a contradiction.

□