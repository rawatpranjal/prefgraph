## An Alternative Approach for Nonparametric Analysis of Random Utility Models ∗

Christopher Turansick

January 27, 2025

## Abstract

We readdress the problem of nonparametric statistical testing of random utility models proposed in Kitamura and Stoye (2018). Although their test is elegant, it is subject to computational constraints which leaves execution of the test infeasible in many applications. We note that much of the computational burden in Kitamura and Stoye's test is due to their test defining a polyhedral cone through its vertices rather than its faces. We propose an alternative but equivalent hypothesis test for random utility models. This test relies on a series of equality and inequality constraints which defines the faces of the corresponding polyhedral cone. Building on our testing procedure, we develop a novel axiomatization of the random utility model.

Keywords : Random Utility, Testing, Revealed Preference

∗ I am thankful to the editor Faruk Gul as well as an anonymous referee for helpful comments which greatly improved the paper. I am also thankful to Peter Caradonna, Christopher P Chambers, Federico Echenique, Yuichi Kitamura, Yusufcan Masatlioglu, Alexandre Poirier, Koji Shirai, Mu Zhang, and seminar participants at EC 24, NASMES 2024, and NBER/NSF/CEME 2023 for helpful discussions and comments during the course of this project.

Department of Decision Sciences and IGIER, Universit` a Bocconi, E-mail: christopher.turansick@unibocconi.it

## 1 Introduction

The random utility paradigm is ubiquitous in modern economics. It is often used to model the choices of a population of rational agents or the repeated choices of a single agent with varying preferences. While the random utility model was initially characterized by Falmagne (1978) and McFadden and Richter (1990), until recently, there has been little work successfully taking these characterizations to real data and testing the random utility hypothesis. Kitamura and Stoye (2018) develop an elegant statistical test of the random utility model, allowing for fully nonparametric testing of unrestricted heterogeneity. However, their test is computationally burdensome and quickly becomes infeasible as the number of available alternatives grows. Kitamura and Stoye (2018) make note of this computational issue. In response to this, Smeulders et al. (2021) prove that the testing procedure of Kitamura and Stoye (2018) is NP-hard and develop computational tools that vastly reduce the time needed to execute the test of Kitamura and Stoye (2018).

In this paper, we take an alternative approach to that of Smeulders et al. (2021). While our goal is the same in that we aim to reduce the computational burden of testing random utility, we instead develop a novel hypothesis test for random utility and show that it offers large improvements over the method of Kitamura and Stoye (2018). To motivate the difference between these two tests, we first note that the set of data points consistent with the random utility model can be written as the convex hull of the data points consistent with deterministically rational choice. This convex set can be represented as either the convex combination of each of these rational choice profiles or it can be represented as the intersection of a finite set of half-spaces. The test of Kitamura and Stoye (2018) utilizes the representation through rational choice profiles while our test utilizes the half-space representation of random utility.

While the half-space representation for random utility is known when the analyst observes choice on every available menu of alternatives (Falmagne, 1978), in general the half-space representation is not known when the analyst observes choice on an arbitrary collection of menus. Our methodology circumvents this issue by using the half-space representation of random utility when every menu is observed. If the data we observe is consistent with random utility, then it admits an extension to data on every menu that is also consistent with random utility. Our methodology introduces slack variables which guarantees that our observed data extends to a full dataset which is consistent with random utility. From a theoretical perspective, we can easily and naively introduce slack variables in order to get a

linear program which characterizes random utility when choice data is limited. However, not every choice of slack variable will be amenable to the current set of econometric tools. A key component of our methodology is that we are able to introduce slack variables which lead to linear programs that are amenable to the econometric tools developed in Fang et al. (2023). Thus our methodology is not only theoretically implementable, but also implementable from an econometric perspective.

In addition to offering computational improvements, our methodology naturally leads to a novel axiomatization of the random utility model when there are unobserved menus. Our new axiom is a statement about when a local form of feasibility extends to a global form of feasibility. These two types of feasibility concern themselves with the assignment of mass to events of the form ' x is chosen from set A ' and the capacity of each set A to contain the mass of these events. Our axiom improves over current axiomatizations of random utility on limited domains as it can be stated without reference to the model. The standard axiom is from McFadden and Richter (1990) and can be recovered by applying the Theorem of the Alternative to the vertex representation (i.e. in terms of deterministic choice functions) of random utility. Our axiom can be recovered by applying the Theorem of the Alternative to the half-space representation of the random utility model.

The rest of this paper is organized as follows. In Section 2 we formally introduce the random utility model as well as discuss the methodologies of Kitamura and Stoye (2018) and Smeulders et al. (2021). In Section 3 we introduce and develop our testing methodology. In Section 4 we present our new axiomatization of the random utility model. Finally, in Section 5 we conclude and offer a discussion of the related literature.

## 2 Random Utility and Testing

Our focus is on the abstract discrete choice setup. This differs from the initial setup of Kitamura and Stoye (2018) (henceforth KS) who focus on random choice from linear pricewealth budgets. Using the results of McFadden (2005), KS show that the testing process in their environment can be reduced to testing a specific version of the abstract discrete choice setup. As such, our focus on the abstract setup is without loss. We will later show how to encode properties such as monotonicity into the abstract setup.

## 2.1 Model

Let X be a finite set of alternatives with typical elements denoted x, y, and z . We assume that an analyst observes choice on some arbitrary collection of subsets of X . We let X ⊆ 2 X \{∅} denote this collection of subsets with typical subsets denoted as A and B . 1 Throughout we will assume that agents have strict preferences over X . 2 In the setup of KS, this is equivalent to assuming single-valued demand. With this assumption in mind, let /follows denote a linear order over X and L ( X ) denote the set of linear orders over X . Our analyst has access to data in the form of a random choice rule.

Definition 2.1. A function p : X × X → R is a random choice rule if it satisfies the following.

- p ( x, A ) ≥ 0 for all x ∈ A
- ∑ x ∈ A p ( x, A ) = 1 for all A ∈ X

In the setup of KS, a random choice rule represents the aggregate choices from a population of agents. Alternatively, a random choice rule can represent the choices of a single agent aggregated across time. The random utility model supposes that there is some distribution over preferences which induces our observed random choice rule. Let ν ∈ ∆( L ( X )) denote a typical probability distribution over linear orders of X .

Definition 2.2. A random choice rule p is stochastically rationalizable if there exists a probability distribution over linear orders ν such that the following holds for all A ∈ X and x ∈ A .

<!-- formula-not-decoded -->

## 2.2 Current Methodology

We now discuss the current methodology for testing the random utility model. Our focus is on the theory and computational burden of the test of KS rather than the statistical properties. Further, our discussion will restrict to the case of idealized data. That is to say,

1 2 X denotes the power set, the collection of each subset, of X .

2 This assumption can be done away with while keeping with our methodology if the analyst has access to a stronger form of data than what we assume here (see Barber´ a and Pattanaik (1986) and Gul and Pesendorfer (2013)).

we assume that p ( x, A ) is the true choice frequency of x from A . In reality, an analyst would observe some number of choices from the choice set A and ˆ p ( x, A ) would be subject to finite sampling error. For a discussion of the statistical properties as well as implementation with real data, we turn the reader to Kitamura and Stoye (2018) and Smeulders et al. (2021).

## 2.2.1 A Conic Approach

To best understand the methodology of KS, we first rewrite the definition of stochastic rationality in matrix form. Consider a matrix M whose rows are indexed by L ( X ), the set of linear orders of X , and whose columns are indexed by pairs of the form ( x, A ), where x ∈ A ∈ X . The element m /follows , ( x,A ) = 1 if x is the maximal element of A according to /follows and m /follows , ( x,A ) = 0 otherwise. Now suppose we can encode a probability distribution over preferences as a vector ν whose indices agree with the rows of M and our random choice rule as a vector p whose indices agree with the columns of M . By doing so, the definition of stochastic rationality can alternatively be given as follows.

Definition 2.3. A random joint choice rule p is stochastically rationalizable if

<!-- formula-not-decoded -->

The first observation that leads to the test of KS is that we can relax the assumption that ν is a probability distribution. Specifically, we need only assume that ν ≥ 0. This is because each row in M encodes a (rational) choice function and ∑ x ∈ A p ( x, A ) = 1 for both choice functions and random choice rules. Formally, this means we can rewrite Equation 2 as

<!-- formula-not-decoded -->

The second observation that leads to the test of KS is that we can turn this existence problem into a quadratic minimization problem. Formally, for a positive definite matrix Ω, there exists a solution to Equation 3 if and only if

<!-- formula-not-decoded -->

The point of transforming the original definition of stochastic rationality into this quadratic form is that the test statistic and the bootstrap technique proposed by KS relies on working with conic shape constraints in this quadratic form.

3 Note that, compared to Kitamura and Stoye (2018), we take the transpose of each matrix.

A key insight from KS related to Equation 3 is that the random utility model, as well as other convex models of stochastic choice, can actually be represented as cones in Euclidean space. From Equation 3, it follows that testing if our data is stochastically rationalizable is equivalent to checking if the data is contained by the following cone.

<!-- formula-not-decoded -->

Equation 5 is known as the vertex representation or V-representation of a polyhedral cone. The V-representation of a cone simply says that a cone is every point that can be generated as a convex combination of each extremal ray of the cone. Each finitely generated cone has an alternative half-space representation or H-representation. The H-representation simply says that a finitely generated cone can always be represented as the intersection of finitely many half-spaces. The equivalence between these two representations is due to the WeylMinkowski Theorem.

Theorem 2.1 (Weyl-Minkowski Theorem) . A subset P of R H is a finitely generated cone

<!-- formula-not-decoded -->

if and only if it is a finite intersection of closed half-spaces

<!-- formula-not-decoded -->

In general, when the H-representation is known, the testing procedure for random utility can be simplified. One could apply tools from the econometric literature on moment inequalities (see Andrews and Soares (2010), Bugni (2010), Canay (2010), and Cox and Shi (2023)). However, the H-representation of random utility varies with X and the exact forms of many of these representations are still unknown (see Gilboa (1990), Gilboa and Monderer (1992), and Cohen and Falmagne (1990)). Notably, the H-representation of random utility is known when X = 2 X \ {∅} and is due to Falmagne (1978). This H-representation is important for our methodology and we will discuss it when we introduce our methodology.

For a moment, consider a general convex model of stochastic choice. If this model forms a simplex, then the M and N matrices in Theorem 2.1 can be chosen to be the same size. This is because simplices have the same number of vertices as they have sides (generating half-spaces). Further, when we are dealing with three or more dimensions, simplices are the

only convex shapes with the same number of vertices and generating half-spaces. Another property of simplices is that every point in a simplex can be written as a unique combination of its vertices. This means that if our convex model of stochastic choice is unidentified, then the M and N matrices in Theorem 2.1 can be of different sizes thus leading to potential computational improvements by considering the H-representation of our model. Since the random utility model is known to be unidentified (Fishburn, 1998; Turansick, 2022) and most of the computation time in the testing procedure of KS comes from the construction of M , it turns out we can improve on the computation time of KS by working with the H-representation of the random utility model.

## 2.2.2 Column (or Row) Generation

An alternative way to reduce the computation time of KS is presented in Smeulders et al. (2021). This method of Smeulders et al. (2021) is called column generation. In order to better understand column generation, we first make a few observations about the identification problem in random utility. As mentioned prior, the random utility model is unidentified when | X | ≥ 4. For datasets which fail to have a unique random utility representation, the support of the representation is also unidentified. 4 For datasets which have a unique random utility representation, when | X | ≥ 4, these representations necessarily are not full support (Turansick, 2022). This means that, from an ex post perspective, if we consider the M matrix in the KS testing procedure, when | X | ≥ 4, there will always be rows that are redundant when we find a rationalizing ν . On the other hand, from an ex ante perspective, no single row of M is redundant as our data can always be induced by a degenerate distribution on the rational type that corresponds to any given row. The column generation procedure of Smeulders et al. (2021) uses the fact that there will always be ex post redundant rational types in the testing procedure of KS.

The column generation procedure begins by guessing that a certain collection of preferences, or rows of M , will not be needed in order to rationalize the data. In doing so, the value of ν ( /follows ) is set equal to zero for each preference /follows in this collection. Then we construct a matrix ¯ M which is the same as matrix M except, for each preference /follows in our guess, it removes the rows associated with /follows . This matrix ¯ M generates an inner approximation of the cone formed by M , so if our random choice rule p lies in the cone formed by ¯ M , it also lies in the cone formed by M . If p does not lie within the cone formed by ¯ M , then we can

4 From Turansick (2022), it is known that, when a dataset has a random utility representation, the representation is unique if and only if the support of the representation is unique.

add one row to ¯ M corresponding to one of the preferences we had assumed to have ν ( /follows ) = 0. We can then check to see if p lies within this new ¯ M and repeat. Smeulders et al. (2021) present a clever pricing problem that allows them to better choose the order which preferences are added back to ¯ M . Since most of the computation time involved in KS comes from construction of M , by iteratively constructing M using the column generation approach, a lot of time is saved in practice.

## 3 New Methodology

In this section, we discuss an alternative methodology to the one proposed in KS for testing the random utility model. This methodology works with the H-representation of a cone, but does not rely on the analyst knowing the H-representation for every X . Our methodology proceeds in three main steps.

1. Find the H-representation of the model when X = 2 X \ {∅} .
2. Perform a change of variables so that each non-negativity constraint of the H-representation can be represented by non-negativity of a single variable.
3. Introduce slack variables which guarantee that the random choice rule on X /negationslash = 2 X \{∅} extends to a random choice rule consistent with the H-representation of the model on 2 X \ {∅} .

We now highlight the key difference between our methodology and the methodology of KS and its impact on implementability. Our methodology works with the H-representation of a model rather than the V-representation. In doing so, we run into theoretical concerns not present in KS. Typically the V-representation of a model is how we define a convex model of stochastic choice. In the case of the random utility model, each data point that is in the convex hull of classically rational choice rules is consistent with random utility. These classically rational choice rules are the vertices of the random utility model and we know what they look like ex ante. However, given a V-representation, it is either a theoretical or computational exercise to find the H-representation of a model. Taking the computational approach to finding an H-representation defeats the purpose of our methodology, so that leaves us with finding H-representations through theoretical means. This is one disadvantage of our methodology when compared to KS.

We now apply our methodology. Recall that the first step in our methodology is to find the H-representation of random utility on 2 X \{∅} . Luckily, this H-representation is already known and is due to Falmagne (1978). Before introducing this representation, we need a bit more notation.

Definition 3.1. The M¨ obius inverse of a random choice rule p : X × 2 X \ {∅} → R is the function q : X × 2 X \ {∅} → R which is defined as follows.

<!-- formula-not-decoded -->

The second line of Equation 8 was introduced by Block and Marschak (1959) and is called the Block-Marschak polynomial. In general, the M¨ obius inverse q ( x, A ) captures how much probability is added to or removed from p ( x, · ) at set A . In terms of the random utility model, these M¨ obius inverse functions have a strong connection to the probability weight put on contour sets.

Theorem 3.1 (Falmagne (1978)) . A distribution over linear orders ν is a random utility representation of a random choice rule p : X × 2 X \ {∅} → R if and only if the following holds for all nonempty A ⊆ X and x ∈ A .

<!-- formula-not-decoded -->

This tells us is that the M¨ obius inverse q ( x, A ) is equal to a probability weight when our random choice rule is stochastically rationalizable. This further means that q ( x, A ) must be non-negative when our random choice rule is stochastically rationalizable. This turns out to be the H-representation of random utility.

Theorem 3.2 (Falmagne (1978)) . A random choice rule p : X × 2 X \ {∅} → R is stochastically rationalizable if and only if q ( x, A ) ≥ 0 for all x ∈ A ⊆ X .

In terms of of the M¨ obius inverse, this means that p is stochastically rationalizable if and only if every set A is contributing some non-negative amount to p ( x, · ). We stated Theorem 3.2 in terms of the M¨ obius inverse q , but recall that, in Equation 8, the M¨ obius inverse is just a linear function of p . Thus, by asking that q ( x, A ) ≥ 0 for all x ∈ A ⊆ X , we are simply

asking that Np ≥ 0 for some matrix N . It should now be apparent that Theorem 3.2 gives us the H-representation of random utility when X = 2 X \ {∅} . It is important to note that Theorem 3.2 does not hold regardless of our domain X . As an example, if X just contains the binary choice sets, having a non-negative M¨ obius inverse is without empirical content. This means that we are unable to simply apply Theorem 3.2 regardless of our domain.

We now move onto the second and third steps of our methodology. As Theorem 3.2 highlights, non-negativity of the M¨ obius inverse functions correspond to the H-representation of random utility. Our goal now is to introduce slack variables that guarantee that our random choice rule p on X extends to 2 X \ {∅} while being consistent with the H-representation of random utility. There are at least two ways we can proceed in introducing slack variables. We can introduce slack variables ˜ p ( x, A ) for sets A /negationslash∈ X that coincide with unobserved choice probabilities. Alternatively, we can introduce slack variables ˜ q ( x, A ) for each set that coincide with the M¨ obius inverse of our hypothetical full domain random choice rule. For now, we focus on slack variables that coincide with choice probabilities. Consider the following linear program.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By naively using ˜ p as our slack variables, we run into a problem. Notably, we have an equality constraint, Equation 10, and a non-negativity constraint, Equation 12, but we have an additional inequality constraint in Equation 11. This additional inequality constraint means that we are unable to directly apply current econometric tools to test the linear program. Fang et al. (2023) develops a bootstrap technique for testing the hypothesis that there exists some x such that

<!-- formula-not-decoded -->

where f ( p ) is some function of the observed data and N is a matrix. Equation 11 is exactly what prevents us from directly applying the test of Fang et al. (2023). Luckily, we can use ˜ q as our slack variables to solve this problem.

One problem arises when we move from ˜ p to ˜ q . When using ˜ p , it is easy to encode that the slack variables induce a random choice rule. Simply ask that ˜ p are non-negative and sum to one at every choice set A . It is less obvious how to guarantee that ˜ q are the M¨ obius

inverse of a full domain random choice rule. Our next result exactly characterizes when this is the case.

Lemma 3.1. 5 A function q : X × 2 X \ {∅} → R is the M¨ obius inverse of some full domain random choice rule p if and only if it satisfies the following conditions.

<!-- formula-not-decoded -->

1. ∑ ∑ 2. ∑ x ∈ X q ( x, X ) = 1
3. ∑ A ⊆ B q ( x, B ) ≥ 0 for all x ∈ A ⊆ X

The first two conditions of Lemma 3.1 guarantee that p sums to one at every choice set. The third condition of Lemma 3.1 guarantees that p is non-negative everywhere. With this in mind, we can now apply step two of our methodology using ˜ q as slack variables. Consider the following linear program.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Above, Equation 14 guarantees that ˜ q is the M¨ obius inverse of some function that agrees with our random choice rule p on sets we observe. Equations 15 and 16 are just the first two conditions of Lemma 3.1 applied at unobserved choice sets. This guarantees that our extended p function induces choice probabilities on unobserved sets. Equation 17 plays two roles in that it is the H-representation of random utility on a full domain and it implies the third condition of Lemma 3.1. The following theorem summarizes our discussion thus far.

Theorem 3.3. The following are equivalent.

1. The random choice rule p : X ×X → R is stochastically rationalizable.
2. There exist variables ˜ p ( x, A ) solving Equations 10-12.

5 We also point out that Kono et al. (2023) contemporaneously developed an analogous result in their Lemma 3.14.

## 3. There exist variables ˜ q ( x, A ) solving Equations 14-17.

Once again, we now take a moment to compare our methodology with that of KS. First, when applying Equations 14-17 and Fang et al. (2023) to real data, the only values that need to be estimated are p ( x, A ) that come from the right hand side of Equation 14. In other words, just as in KS, we only need to estimate choice probabilities; each other variable in our testing procedure is constructed. Second, we can now begin to compare the computational burden of each testing procedure. While the bootstrapping procedure in KS and Fang et al. (2023) differ, in both KS and our testing procedure, almost all of the computation time comes from construction of a matrix; the M matrix in the case of KS and an N matrix which encodes Equations 14-17 in our case. 6 The M matrix of KS has one column for each pair ( x, A ) with x ∈ A ∈ X and one row for each linear order over X . This means that the number of rows in M grows at a rate of | X | !. Our N matrix has one column for each pair ( x, A ) with x ∈ A ∈ 2 X \ {∅} and one row for each condition in Equations 14-17.

Proposition 3.1. The number of rows in N grows at a rate of | X | 2 | X |-1 -∑ A /negationslash∈X ( | A | -1) .

When | X | ≤ 4, M has fewer rows than N , but when | X | ≥ 5, N always has fewer rows than M . In Table 1 we calculate the number of rows in M and N for a few sizes of X when X = 2 X \{∅} as this is the worst case for our N matrix. As Table 1 points out, as | X | grows, the number of rows in M becomes vastly larger than the number of rows in N .

Table 1: The number of rows in the M and N matrices are given as a function of | X | under the assumption that X = 2 X \ {∅} . Notably, in the case of | X | = 15, the number of rows of M given in this table is an underapproximation.

|   &#124; X &#124; | M rows          | N rows    |
|-------------------|-----------------|-----------|
|                 3 | 6               | 12        |
|                 4 | 24              | 32        |
|                 5 | 120             | 80        |
|                10 | 3 , 628 , 800   | 5120      |
|                15 | ∼ 1 . 3 × 10 12 | 245 , 760 |

Wenow take a moment to discuss how the column generation procedure of Smeulders et al. (2021) can be applied in our methodology. Recall that the column generation procedure begins by guessing that some collection of our choice variables are equal to zero. In our setup, these choice variables correspond to ˜ q . When X = 2 X \ {∅} , each ˜ q is constructed directly

6 Formally, we can represent the existence of variable q satisfying Equations 14-17 as the existence of a variable satisfying Nq = l for some matrix N and a vector l .

from the data, and thus the exact value of each ˜ q is directly observable. This means that the column generation procedure offers no improvements in this case. However, when X is missing sets, there is room to apply the column generation procedure. Notably, when X is not complete, we are unable to directly construct ˜ q from the data. This means that it is possible that some of our slack variables can be chosen to be equal to zero. Thus in this case, the column generation procedure can offer computational improvement when added onto our methodology. However, unlike in the case of the KS testing procedure, there is no ex ante guarantee that we can always guess some ˜ q to be equal to zero.

## 3.1 Encoding Monotone Choice

Thus far we have developed a methodology for testing random utility on an abstract discrete choice domain. Until now, we have ignored two components of the problem present in the original work of KS. First, in KS the choice data is not from an abstract discrete choice domain but rather comes from choice frequencies on linear price-wealth budgets. Second, in KS utility functions are restricted to be monotone with respect to the greater than or equal to ordering on R n . In this section, we discuss how to incorporate and deal with these two problems using our methodology.

We first discuss going between the linear budget domain and abstract discrete choice domain. As is the case in KS and McFadden (2005), we will assume we are working with monotone utility functions in the linear budget domain. A result of this monotonicity assumption is that all observed choice should occur on our budget lines. If we observe choice strictly within our budget, then we know that the consumer has money left that they can spend on more of a good to obtain a higher utility. It then follows from an observation made in McFadden (2005) and used in KS that we need only focus on the partition of our budgets formed by points of intersection. McFadden (2005) points out that the entire content of random monotone utility with linear budgets is captured by agents' choices on patches. Suppose we have two linear budgets with a single point of intersection, as is the case in Figure 1. Then each of these linear budgets can be partitioned into three components: the point of intersection, 'above' the point of intersection, and 'below' the point of intersection. Each of these three components correspond to a patch in the language of McFadden (2005) and KS. In Figure 1, if we assume there is no choice at the point of intersection, then the set of patches we are left with is given by { w, x, y, z } . This set of patches then corresponds to our choice set X in the abstract discrete choice environment. This construction scales beyond

Figure 1: Here we capture two linear budgets over the goods a 1 and a 2 . These budgets are labeled as B 1 and B 2 and have a single point of intersection. By assuming that no choice occurs at the point of intersection, we are left with { w, x, y, z } as the set of patches.

![Image](data:image/png;base64,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)

the case of two budgets with a single point of intersection.

Our ability to go from the linear budget setup to discrete choice on patches relies on the assumption of monotone utility. Further, the assumption of monotone utility is natural in the linear budget setup of KS. However, the assumption of monotone utility imposes further testable content beyond that of the abstract random utility model. Notably, in the case considered in Figure 1, the H-representation is known. It asks that p ( y, { x, y } ) + p ( z, { w, z } ) ≤ 1. Unfortunately, the H-representation for the linear budget setup becomes increasingly more complicated as the set of budgets and the set of intersections grow. Recall that our methodology works with the H-representation when X = 2 X \ {∅} . This type of data can never be observed in the linear budget setup. That being said, our methodology simply asks if the data we observe can be extended to a full domain H-representation. With this in mind, consider Figure 1 and suppose an agent is choosing from the set { w, y } . The patch y is dominated by w in the standard monotone ordering of R 2 . As such, an agent with monotone utility will never choose y when w is available. This observation, initially due to Kashaev et al. (2023), turns out to be the only further testable content of monotone random utility beyond that of abstract random utility.

Definition 3.2. For X partially ordered by /triangleright , we say that a random choice rule p : X × 2 X \ {∅} → [0 , 1] is stochastically rationalizable by monotone utilities if it is stochastically

rationalizable and there exists a rationalizing distribution ν such that ν ( /follows ) &gt; 0 and x /triangleright y imply x /follows y .

Definition 3.3. We say that a random choice rule p is monotone with respect to a partial order /triangleright if x, y ∈ A and x /triangleright y implies that p ( y, A ) = 0.

Theorem 3.4 (Kashaev et al. (2023)) . Suppose X is partially ordered by /triangleright . A random choice rule p : X × 2 X \ {∅} → [0 , 1] is stochastically rationalizable by monotone utilities if and only if q ( x, A ) ≥ 0 for all x ∈ A ⊆ X and p is monotone with respect to /triangleright .

Theorem 3.4 gives us the H-representation we are looking for on a full domain. However, in our Theorem 3.3, we rely on working only with the M¨ obius inverse while Theorem 3.4 utilizes both the base random choice rule and the M¨ obius inverse. Our goal now is to transform the monotonicity condition on a random choice rule into a condition on its M¨ obius inverse. This turns out to be relatively easy to do given that we already impose that q ( x, A ) ≥ 0 in order to get stochastic rationality. For each x ∈ X , let U ( x ) denote the set of alternatives dominating x according to /triangleright . Then for each x with nonempty U ( x ), we can encode monotonicity with respect to /triangleright using the following equality constraint.

/negationslash

<!-- formula-not-decoded -->

/negationslash

We can then use Equation 18 along side our prior characterizations of stochastic rationality to get a characterization of stochastic rationality by monotone utilities.

## Proposition 3.2. The following are equivalent.

1. A random choice rule p : X ×X → [0 , 1] is stochastically rationalizable by monotone utilities.
2. There exists solutions to Equations 14-17 and Equation 18.

Since Equation 18 is an equality constraint, we are able to use the econometric tools of Fang et al. (2023) to test for random monotone utility. In terms of the resulting matrix, we only need to add a single row for each alternative x with nonempty U ( x ). For the growth rate of the number of rows in N , the additional monotonicity constraints are of a lower order, being bounded above by | X | -1. This means that the rate in Proposition 3.1 is still approximately correct and is off by at most | X | -1 in the case of random monotone utility.

## 4 Axiomatics

Thus far, our focus has been on developing an alternative testing methodology that offers computational improvements over the testing procedure of KS. In this section, we turn our attention to axiomatically characterizing the random utility model using the methodology we developed in the prior section. The main result in this section is a new characterization of random utility on a limited domain. The current standard characterization of random utility on a limited domain is due to McFadden and Richter (1990).

Theorem 4.1 (McFadden and Richter (1990)) . A random choice rule p : X × X → R is stochastically rationalizable if and only if for any finite sequence { ( x i , A i ) } n i =1 with x i ∈ A i ∈ X the following holds.

<!-- formula-not-decoded -->

We now discuss the relationship between the methodology of KS and Theorem 4.1. As pointed out earlier, the methodology of KS relies on the V-representation of the random utility model. The characterization in Theorem 4.1 relies on the V-representation of random utility in the sense that Equation 19 follows from applying the Theorem of the Alternative to the V-representation of random utility (see Border (2013) and Border (2007) for references). As a result of working with the V-representation and since the V-representation of random utility is itself the model representation of random utility, Equation 19 makes explicit use of the underlying random utility model. That is to say, in order to state Equation 19, we must make reference to preferences.

Recall that our methodology works with the H-representation of random utility. We get our new axiom by effectively applying the Theorem of the Alternative to the H-representation of random utility. By working with the H-representation, our resulting axiom makes no reference to preferences or the random utility model. Our characterization relies on two supplemental functions.

Definition 4.1. A function c : 2 X → R is a capacity if c ( ∅ ) = 0.

Definition 4.2. A function a : X ×X → R is an assignment .

In order to best interpret the role of capacities and assignments, we introduce the follow-

ing definition. 7

Definition 4.3. Given a random choice rule p , an assignment and capacity pair ( a, c ) is feasible if the following inequality holds.

<!-- formula-not-decoded -->

The role of an assignment a is to assign some mass to the event that x is chosen from A . When combined with the probability that x is chosen from A , p ( x, A ), this mass is given by a ( x, A ) p ( x, A ). Each set A has a capacity c ( A ). This capacity must be more than the total mass put on events of the form ' x is chosen from B ' for each B ⊆ A . Feasibility simply asks that the total weight put on choosing some element from some set is less than the capacity of X , the total capacity of our environment. Our characterization relies on a second type of feasibility.

Definition 4.4. An assignment and capacity pair ( a, c ) is locally feasible if for each ( x, A ) with x ∈ A ∈ 2 X \ {∅} the following inequality holds.

<!-- formula-not-decoded -->

Local feasibility is a local condition in the sense that it states that the total amount of capacity gained by going from A \{ x } to A must be more than the total amount of assigned mass to x being chosen in subsets of A . Note that, unlike feasibility, local feasibility does not depend on the random choice rule p and the set of locally feasible assignment and capacity pairs can be defined independently of the observed data. Our characterization of stochastic rationalizability is a condition about when local feasibility implies feasibility.

Theorem 4.2. Given a random choice rule p : X ×X → R , the following are equivalent.

- p is stochastically rationalizable.
- Every locally feasible assignment and capacity pair ( a, c ) is also feasible given p .

Our main innovation over Theorem 4.1 is that we use Lemma 3.1 to rewrite the stochastic

7 Note that our definitions of capacity and assignment differ from those used in cooperative game theory. In cooperative game theory, capacities are typically monotone and assignments typically assign a value to each alternative/agent rather than one value to an alternative for each set containing the alternative.

rationality linear program without reference to any preferences.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equations 22-25 bear an obvious resemblance to Equations 14-17. The notable difference is that we ask that Equation 15 holds at every choice set rather than simply at unobserved choice sets. In terms of our characterization, this is what gives us that capacity functions are defined on every choice set and the right hand side of Equation 21. Once we have Equations 22-25, we simply apply the Theorem of the Alternative and do some minor manipulation in order to get Theorem 4.2.

## 5 Discussion

In this paper we propose a new procedure for testing nonparametric models of random utility. In this test as well as the test of Kitamura and Stoye (2018), much of the computation time is due to calculation of a given matrix. We show that our test will generally lead to a much smaller matrix when compared to the matrix of KS and thus offer large computational improvements over the test of KS.

While our focus in this paper is on testing the random utility model, our testing procedure can be modified for other convex models of choice. 8 Recall that our methodology proceeds in three steps.

1. Find the H-representation of the model when X = 2 X \ {∅} .
2. Perform a change of variables so that each non-negativity constraints of the H-representation can be represented by non-negativity of a single variable.

8 By convex model of choice, we mean any model of choice where the set of data points consistent with the model is given by a convex set. Frequently it is the case that the model is defined by the extreme points of this convex set.

3. Introduce slack variables which guarantee that the random choice rule on X /negationslash = 2 X \{∅} extends to a random choice rule consistent with the H-representation of the model on 2 X \ {∅} .

The first step of this testing procedure extends naturally to any convex model of choice. It becomes a bit more difficult when translating the second and third steps to other convex models. As pointed out by Equations 10-12 and the discussion thereafter, we are unable to apply the econometric tools of Fang et al. (2023) if we naively introduce the wrong slack variables in step two of our methodology. The reason why we are able to proceed to step three using Equations 14-17 is that Lemma 3.1 characterizes random choice rules in terms of the variables induced by the H-representation of the random utility model. In other words, by the Weyl-Minkowski Theorem, there exists some matrix H such that a random choice rule is consistent with our convex model of choice if and only if Hp ≥ 0. This H matrix is a linear transformation of choice probabilities. In our case, this H matrix induced our M¨ obius inverse function. More generally, this H matrix simply induces some variable q ( H,p ) which depends on our observed random choice rule as well as the matrix itself. To go between steps two and three, we need to do two things. First, we need to characterize random choice rules in terms of these q ( H,p ) variables. This characterization will typically consist of equality conditions which correspond to choice probabilities summing to one and inequality conditions which corresponds to probabilities being non-negative. Second, we need to check if non-negativity of our q ( H,p ) variables implies the non-negativity conditions of the random choice rule characterization. If it does, we are left with a collection of equality constraints and q ( H,p ) ≥ 0 as the only set of inequality constraints. The fact that q ( H,p ) ≥ 0, or q ( x, A ) ≥ 0 in our case, is the only inequality constraint is exactly what lets us use the econometric tools of Fang et al. (2023).

## 5.1 Related Literature

Our paper is related to two primary strands of literature. First, our paper is related to the strand of literature started by Kitamura and Stoye (2018) which focuses on hypothesis testing the random utility model and other similar models. Following the lead of Kitamura and Stoye (2018), Smeulders et al. (2021) also studies the problem of hypothesis testing the random utility model and, similar to us, focuses on the problem of computational implementability. However, unlike us, they take the base testing procedure of Kitamura and Stoye (2018) as given and try to significantly improve the computation

time of that test. In other words, they continue working with the V-representation of the random utility model while we work with the H-representation. In a working paper, Forcina and Dardanoni (2023) studies the bootstrapping procedure used in the hypothesis test of Kitamura and Stoye (2018). The authors propose two alternative bootstrapping techniques and compare the accuracy of these techniques to the one proposed in Kitamura and Stoye (2018) using simulations. Fang et al. (2023) proposes an alternative bootstrap technique which can be used to test the hypothesis of Kitamura and Stoye (2018). There have also been a series of papers using and extending the techniques proposed in Kitamura and Stoye (2018) in order to study other related models. Deb et al. (2023) extend the techniques of Kitamura and Stoye (2018) in order to test a general model of preferences over prices as well as do welfare analysis. Kashaev et al. (2023) extend the techniques of Kitamura and Stoye (2018) in order to test a dynamic version of the random utility model allowing for correlation of preferences over time. Finally, Dean et al. (2022) also use the techniques of Kitamura and Stoye (2018) to develop a better test, in the sense of power, for choice overload.

Our paper is also related to the literature which axiomatically studies the random utility model. Falmagne (1978) is the first to characterize the random utility and does so by asking that the M¨ obius inverse of choice probabilities be non-negative. Fiorini (2004) offers an alternative proof of this result using graph theoretic techniques. Monderer (1992) provides an alternate proof of this result using methods from cooperative game theory. Cohen (1980) considers an extension of the result of Falmagne (1978) to an infinite domain. Nandeibam (2009) provides a different characterization of random utility using positive linear functionals. McFadden and Richter (1990) offers a characterization of random utility when the choice domain is incomplete. Stoye (2019) offers a short proof of this result using tools from convex analysis. McFadden (2005) offers an extension of this result to an infinite domain under some regularity conditions. Recently, Gonczarowski et al. (2023) extends this result to an infinite domain without any regularity conditions. Clark (1996) offers an alternative characterization of random utility in the case of an incomplete domain using DeFinetti's coherency axiom. Kashaev et al. (2023) uses techniques developed to study quantum entanglement in order to offer a characterization of dynamic separable random utility on a limited domain. Kono et al. (2023) study and axiomatize the random utility model when the choice probabilities of a collection of goods are unobservable at every choice set. Lastly, Koida and Shirai (2024) studies the the random monotone utility hypothesis in the price-wealth budget domain. They are able to find the H-representation of the model which relies on the ordering of the underlying environment.

## A Preliminary Results

## A.1 Proof of Lemma 3.1

This proof proceeds in two steps. First we show the equivalence between the M¨ obius inverse of a function satisfying the first condition in Lemma 3.1 and that function being set constant.

Definition A.1. A function p : 2 X \ {∅} → R is set constant if for all A, B ∈ 2 X \ {∅} we have that ∑ x ∈ A p ( x, A ) = ∑ y inB p ( y, B ).

To note, henceforth, if the M¨ obius inverse of a function p satisfies the first condition of Lemma 3.1, then we will say that the function p satisfies inflow equals outflow. After doing this, the rest of the proof amounts to noting that the second and third conditions are direct translations of ∑ x ∈ X p ( x, X ) = 1 and p ( x, A ) ≥ 0 into statements about the M¨ obius inverse. We begin by showing the equivalence of p satisfying inflow equals outflow and p being set constant. We start with the necessity of inflow equals outflow. Consider a function f with M¨ obius inverse g such that f is set constant. We proceed via induction on the size of the complement of A . For the base case, let A = X \ { x } . Observe that f ( x, X ) = g ( x, X ) We have the following.

<!-- formula-not-decoded -->

Above, the first equality holds by the definition of M¨ obius inverse. The second equality holds from f being set constant. The third equality follows after collecting like terms. This shows that the base case of inflow equals outflow holds. Now assume that inflow equals outflow holds for all B with | X \ B | &lt; n . Let A be such that | X \ A | = n .

<!-- formula-not-decoded -->

Above, the first equality holds by the definition of M¨ obius inverse. The second equality just adds zero. The third equality holds as g ( x, X ) = f ( x, X ) and because f is set constant. The fourth equality holds by the induction hypothesis. The fifth equality follows from combining like terms. Thus the above string of equalities show that inflow equals outflow is necessary. We now show sufficiency. Now suppose f satisfies inflow equals outflow. Consider some A /subsetnoteql X .

<!-- formula-not-decoded -->

The first equality above holds due to the definition of M¨ obius inverse. The second equality just adds zero. The third equality holds as f ( x, X ) = g ( x, X ). The fourth equality holds by inflow equals outflow. The fifth equality follows from combining like terms. By inflow equals outflow, we know that ∑ x ∈ A g ( x, A ) = ∑ z ∈ X \ A g ( z, A ∪ { z } ). This means that the above string of equalities gives us that ∑ x ∈ A f ( x, A ) = ∑ x ∈ X f ( x, X ). Since A is arbitrary, this tells us that f is set constant.

Now we return to the case of choice probabilities. Recall that q ( x, X ) = p ( x, X ). This means that asking ∑ x ∈ X p ( x, X ) = 1 is equivalent to asking that ∑ x ∈ X q ( x, X ). Further, asking that p ( x, A ) ≥ 0 is equivalent to asking that ∑ A ⊆ B q ( x, B ) ≥ 0 by the definition of the M¨ obius inverse. It then follows that asking for the three conditions in Lemma 3.1 to hold is equivalent to asking that the following conditions hold.

1. ∑ x ∈ A p ( x, A ) = ∑ y ∈ B p ( y, B ) for all A, B ∈ 2 X \ {∅}
2. ∑ x ∈ X p ( x, X ) = 1
3. p ( x, A ) ≥ 0 for all x ∈ A ⊆ X

The above three conditions define a random choice rule. Thus Lemma 3.1 holds.

## A.2 Proof of Lemma A.1

We now prove a weaker version of Theorem 3.3 that will be useful in the proof of Theorem 3.3 and 4.2.

Lemma A.1. The following statements are equivalent.

1. A random choice rule p : X ×X → [0 , 1] is stochastically rationalizable.
2. There exists a solution to Equations 22-25.

We now proceed with our proof of Lemma A.1. We proceed with sufficiency of our second condition. Observe the following. Equations 23-25 imply that ˜ q is the M¨ obius inverse of some full domain random choice rule. Further, Equation 25 implies that this full domain random choice rule is stochastically rationalizable. Lastly, Equation 22 implies that the full domain random choice rule induced by ˜ q agrees with our observed data on sets we observe. Thus our second condition implies stochastic rationalizability. Now we proceed with necessity of the second condition. Observe that if our random choice rule p is stochastically rationalizable, then it admits an extension to a full domain (i.e. 2 X \ {∅} that is also stochastically rationalizable. A full domain random choice rule is an extension of our observed random choice rule if and only if its M¨ obius inverse satisfies Equation 22. Further, a full domain random choice rule is stochastically rationalizable if and only if its M¨ obius inverse satisfies Equation 25. Finally, since Equation 25 implies the third condition in Lemma 3.1, if the

M¨ obius inverse of a function satisfies Equations 23-25 then the function is a random choice rule by Lemma 3.1. Thus Equations 22-25 are necessary for stochastic rationality.

## B Omitted Proofs

## B.1 Proof of Theorem 3.3

We begin with the equivalence between stochastic rationalizability and Equations 10-12. Equations 10 and 12 is equivalent to asking that there is a full domain extension of our random choice rule. By Theorem 3.2, we know that a full domain random choice rule is stochastically rationalizable if and only if ∑ A ⊆ B ( -1) | B \ A | p ( x, B ) ≥ 0 for all x ∈ A ⊆ X . Thus asking that there exist a solution to Equations 10-12 is equivalent to asking that there is some full domain extension of our random choice rule that satisfies the conditions of Theorem 3.2. Thus stochastic ratioanlizability is equivalent to there be a solution to Equations 10-12.

We now move on to the equivalence between stochastic rationalizability and the existence of a solution to Equations 14-17. We proceed by showing an equivalence between Equations 14-17 and Equations 22-25. As the constraints in Equations 14-17 are a subset of the constraints in Equations 22-25, it follows immediately that if Equations 22-25 have a solution then Equations 14-17 have a solution. By Lemma A.1, stochastic rationalizability is equivalent to there being a solution to Equations 22-25, and so we are done with this direction. We now show the other direction. Let q be a solution to Equations 22-25. We proceed by induction on A .

The base case of our induction is when A = X . Note that either X ∈ X or x /negationslash∈ X . If X /negationslash∈ X , then Equations 14-17 coincide with Equations 22-25 at X . If X ∈ X , then we observe choice probabilities at X , and so ˜ q ( x, X ) ≥ 0 for all x ∈ X as p ( x, X ) = ˜ q ( x, X ) by Equation 14. Since probabilities sum to one, this gives us ∑ x ∈ X p ( x, X ) = ∑ x ∈ X ˜ q ( x, X ) = 1 which is exactly Equation 24. Thus Equations 22-25 hold at X . Now fix a set A /subsetnoteql X and suppose that Equations 22-25 hold for every set B such that A /subsetnoteql B . This is our induction hypothesis. There are two cases; either A ∈ X or A /negationslash∈ X . If A ∈ X , then we know that ∑ x ∈ A p ( x, A ) = 1 as choices are observed at A . Further, by our induction hypothesis ∑ B ⊆ B ′ ˜ q ( x, B ′ ) = p ( x, B ). It then follows that ˜ q is the M¨ obius inverse of p on the domain { B | A ⊆ B } . This gives us that p ( x, A ) = ∑ A ⊆ B ˜ q ( x, B ). As ˜ q ( x, A ) ≥ 0 is assumed in Equation 17, all we need to show is that Equation 23 holds at A . Since p ( · ) is set constant

on { B | A ⊆ B } , it follows from Lemma 3.1 that Equation 23 holds at A . Thus, if A ∈ X , then Equations 22-25 hold at set A . Now suppose that A /negationslash∈ X . When A /negationslash∈ X , Equations 14-17 coincide with Equations 22-25 at set A . Thus by induction, if Equations 14-17 hold, then Equations 22-25 hold.

## B.2 Proof of Proposition 3.1

We begin with the case that X = 2 X \ {∅} . In this case, only Equation 14 is encoded in our N matrix. In this case, the matrix N has one row for each pair ( x, A ) with x ∈ A ⊆ X . This is given by ∑ | X | i =1 i ( | X | i ) = | X | 2 | X |-1 . Now consider the general case. In the general case, for each A /negationslash∈ X , we remove one case of Equation 14 for each x ∈ A and replace it with one case of Equation 15 (or Equation 16 if A = X ). This change from the full domain case is equal to | A | -1, thus giving us | X | 2 | X |-1 -∑ A /negationslash∈X ( | A | -1) and so we are done.

## B.3 Proof of Proposition 3.2

/negationslash

From Theorem 3.3, we already know that stochastic rationalizability is equivalent to the existence of solutions to Equations 14-17. All that is left to show is that Equation 18 is equivalent to rationalizability by monotone utilities in the case of stochastic rationalizability. By Theorem 3.4, we know that rationalizability by monotone utilities is equivalent to p ( x, A ) = 0 whenever x contains an alternative y such that y dominates x in the underlying ordering of X . This is equivalent to ∑ A ⊆ A q ( x, B ) = 0 for the chosen ( x, A ). Further, note that if A contains a y dominating x , then every superset of A contains a y dominating x . Since we are in the case of stochastic rationalizability, this means that q ( x, B ) ≥ 0 for all ( x, B ). By the prior logic, this means p ( x, A ) = 0 for A with y dominating x is equivalent to q ( x, B ) = 0 for all A ⊆ B . It then immediately follows that q ( x, B ) = 0 for all B with y dominating x . In the case of q ( x, A ) ≥ 0, this is equivalent to ∑ A : x ∈ A,A ∩ U ( x ) = ∅ q ( x, A ) = 0 ∀ x such that U ( x ) = ∅ , and so we are done.

## B.4 Proof of Theorem 4.2

Let N ( x, A ) = {/follows∈ L ( X ) | x /follows y ∀ y ∈ A \{ x }} . A random choice rule on X is stochastically rationalizable if and only if there exists ν ∈ ∆( L ( X )) such that p ( x, A ) = ∑ /follows∈ N ( x,A ) ν ( /follows )

/negationslash

for all x ∈ A ∈ X . This is equivalent to the existence of such a ν and the existence of choice probabilities p ( y, B ) for each B ∈ (2 X \ {∅} ) \ X such that p ( x, A ) = ∑ /follows∈ N ( x,A ) ν ( /follows ) for all x ∈ A ∈ 2 X \ {∅} . By Theorem 3.2, this is equivalent to the existence of choice probabilities p ( y, B ) for each B ∈ (2 X \ {∅} ) \ X such that q ( x, A ) ≥ 0 for each x ∈ A ∈ 2 X \ {∅} . By Lemma 3.1, this is equivalent to the existence of q ( x, A ) satisfying the conditions of Lemma A.1.

We now construct the matrix form of this linear program. Consider a matrix D whose columns are indexed by ( x, A ) for each A ∈ 2 X \ {∅} and each x ∈ A and whose rows are indexed by ( y, B ) for each B ∈ X and y ∈ B . The entry d ( y,B ) , ( x,A ) = 1 if B ⊆ A and x = y and is equal to zero otherwise. D encodes that our unobserved q function must induce our observed choice probabilities. Let P be a column vector indexed by ( y, B ) for each B ∈ X and y ∈ B . Entry p ( y,B ) is equal to p ( y, B ). Consider a matrix E whose columns are indexed by ( x, A ) for each A ∈ 2 X \ {∅} and each x ∈ A and whose rows are indexed by B ∈ 2 X \ { X, ∅} . The entry e B, ( x,A ) is given as follows.

<!-- formula-not-decoded -->

E encodes that our unobserved q satisfy inflow equals outflow. Consider a row vector F whose elements are indexed by ( x, A ) for each A ∈ 2 X \ {∅} and each x ∈ A . The element f ( x,A ) is equal to one if A = X and equal to zero otherwise. F encodes that our unobserved q satisfy ∑ x ∈ X q ( x, X ). To be stochastically rationalizable, we must impose q ≥ 0 which implies the last condition of Lemma 3.1. Thus the linear program we have constructed looks as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Farkas's Lemma, (see Theorem 34 in Border (2013) for a reference), there exists a solution to this linear program if and only if there does not exist a solution r ∈ R M to the following

linear program.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Writing out Equation 29 gives us the following.

<!-- formula-not-decoded -->

Above, r ( X ) is the element of r which is associated with the vector F . As r ( X ) can be any real number, we can rewrite Equation 30 as follows.

<!-- formula-not-decoded -->

For a given ( x, A ), Equation 28 can be written as follows.

<!-- formula-not-decoded -->

Above, in the case that A \ { x } = ∅ , we define r ( ∅ ) = 0. Further, r ( X ) shows up as -r ( X ) as a consequence of our transformation of Equation 30 into Equation 31. The negation of the existence of some r solving Equations 31 and 32 is equivalent to the negation of every locally feasible assignment and capacity pair ( a, c ) being feasible. Thus it follows that the condition of Theorem 4.2 holds if and only if p is stochastically rational.

## References

- Andrews, D. W. and G. Soares (2010): 'Inference for parameters defined by moment inequalities using generalized moment selection,' Econometrica , 78, 119-157.
- Barber´ a, S. and P. K. Pattanaik (1986): 'Falmagne and the Rationalizability of Stochastic Choices in Terms of Random Orderings,' Econometrica , 54, 707-715.
- Block, H. D. and J. Marschak (1959): 'Random Orderings and Stochastic Theories of Response,' Tech. rep., Cowles Foundation for Research in Economics, Yale University.

- Border, K. (2007): 'Introductory notes on stochastic rationality,' California Institute of Technology.
- --- (2013): 'Alternative Linear Inequalities,' California Institute of Technology.
- Bugni, F. A. (2010): 'Bootstrap inference in partially identified models defined by moment inequalities: Coverage of the identified set,' Econometrica , 78, 735-753.
- Canay, I. A. (2010): 'EL inference for partially identified models: Large deviations optimality and bootstrap validity,' Journal of Econometrics , 156, 408-425.
- Clark, S. A. (1996): 'The random utility model with an infinite choice space,' Economic Theory , 7, 179-189.
- Cohen, M. and J.-C. Falmagne (1990): 'Random utility representation of binary choice probabilities: A new class of necessary conditions,' Journal of Mathematical Psychology , 34, 88-94.
- Cohen, M. A. (1980): 'Random utility systems-the infinite case,' Journal of Mathematical Psychology , 22, 1-23.
- Cox, G. and X. Shi (2023): 'Simple adaptive size-exact testing for full-vector and subvector inference in moment inequality models,' The Review of Economic Studies , 90, 201-228.
- Dean, M., D. Ravindran, and J. Stoye (2022): 'A Better Test of Choice Overload,' arXiv preprint arXiv:2212.03931 .
- Deb, R., Y. Kitamura, J. K. Quah, and J. Stoye (2023): 'Revealed price preference: theory and empirical analysis,' The Review of Economic Studies , 90, 707-743.
- Falmagne, J.-C. (1978): 'A Representation Theorem for Finite Random Scale Systems,' Journal of Mathematical Psychology , 18, 52-72.
- Fang, Z., A. Santos, A. M. Shaikh, and A. Torgovitsky (2023): 'Inference for Large-Scale Linear Systems With Known Coefficients,' Econometrica , 91, 299-327.
- Fiorini, S. (2004): 'A Short Proof of a Theorem of Falmagne,' Journal of Mathematical Psychology , 48, 80-82.
- Fishburn, P. C. (1998): 'Stochastic Utility,' in Handbook of Utility Theory , ed. by S. Barbera, P. Hammond, and C. Seidl, Kluwer Dordrecht, 273-318.

- Forcina, A. and V. Dardanoni (2023): 'Methods for Testing the Random Utility Model,' SSRN Preprint .
- Gilboa, I. (1990): 'A necessary but insufficient condition for the stochastic binary choice problem,' Journal of Mathematical Psychology , 34, 371-392.
- Gilboa, I. and D. Monderer (1992): 'A Game-Theoretic Approach to the Binary Stochastic Choice Problem,' Journal of Mathematical Psychology , 36, 555-572.
- Gonczarowski, Y. A., S. D. Kominers, and R. I. Shorrer (2023): 'To Infinity and Beyond: A General Framework for Scaling Economic Theories,' arXiv preprint arXiv:1906.10333 .
- Gul, F. and W. Pesendorfer (2013): 'Random Utility Maximization with Indifference,' Tech. rep., Working Paper.
- Kashaev, N., V. H. Aguiar, M. Pl´ avala, and C. Gauthier (2023): 'Dynamic and Stochastic Rational Behavior,' arXiv preprint .
- Kitamura, Y. and J. Stoye (2018): 'Nonparametric Analysis of Random Utility Models,' Econometrica , 86, 1883-1909.
- Koida, N. and K. Shirai (2024): 'A dual approach to nonparametric characterization for random utility models,' arXiv preprint arXiv:2403.04328 .
- Kono, H., K. Saito, and A. Sandroni (2023): 'Axiomatization of Random Utility Model with Unobservable Alternatives,' arXiv preprint arXiv:2302.03913 .
- McFadden, D. and M. K. Richter (1990): 'Stochastic Rationality and Revealed Stochastic Preference,' Preferences, Uncertainty, and Optimality, Essays in Honor of Leo Hurwicz, Westview Press: Boulder, CO , 161-186.
- McFadden, D. L. (2005): 'Revealed Stochastic Preference: A Synthesis,' Economic Theory , 26, 245-264.
- Monderer, D. (1992): 'The Stochastic Choice Problem: A Game-Theoretic Approach,' Journal of Mathematical Psychology , 36, 547-554.
- Nandeibam, S. (2009): 'On probabilistic rationalizability,' Social Choice and Welfare , 32, 425-437.

- Smeulders, B., L. Cherchye, and B. De Rock (2021): 'Nonparametric Analysis of Random Utility Models: Computational Tools for Statistical Testing,' Econometrica , 89, 437-455.
- Stoye, J. (2019): 'Revealed Stochastic Preference: A one-paragraph proof and generalization,' Economics Letters , 177, 66-68.
- Turansick, C. (2022): 'Identification in the random utility model,' Journal of Economic Theory , 203, 105489.