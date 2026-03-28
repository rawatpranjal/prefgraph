
## Bibliografi  sche Informationen der Deutschen Nationalbibliothek

Die Deutsche Bibliothek verzeichnet diese Publikation in der deutschen Nationalbibliografi  e; detaillierte bibliografi  sche Daten sind im Internet über: http://dnb.d-nb.de abrufb   ar.

## Jan Heufer and Per Hjertstrand 1

## Consistent Subsets - Computationally Feasible Methods to Compute the Houtman-Maks-Index

## Abstract

We provide two methods to compute the largest subset of a set of observations that is consistent with the Generalised Axiom of Revealed Preference. The algorithm provided by Houtman and Maks (1985) is not computationally feasible for larger data sets, while our methods are not limited in that respect. The fi  rst method is a variation of Gross and Kaiser's (1996) approximate algorithm and is only applicable for two-dimensional data sets, but it is very fast and easy to implement. The second method is a mixed -integer linear programming approach that is slightly more involved but still fast and not limited by the dimension of the data set.

JEL Classifi  cation: C14, D11, D12

Keywords:  Demand  theory;  e/uniFB03 ciency;  nonparametric  analysis;  revealed  preference; utility maximisation

December 2014

1  Jan  Heufer, TU Dortmund; Per Hjertstrand, Research Institute of Industrial Economics, Stockholm. - Access to data collected and previously used by Syngjoo Choi, Raymond Fisman, Douglas Gale, and Shachar Kariv is gratefully acknowledged. - All correspondence to: Jan Heufer, TU Dortmund University, Department of Economics and Social Science, 44221 Dortmund, Germany, e-mail: jan.heufer@tu-dortmund.de

## 1 introduction

When consumer choice data violates utility maximising behaviour it is often desirable to know by 'how much' the observed choices deviate from utility maximisation. Houtman and Maks (1985) proposed to measure the degree of inconsistency as the maximal number of observations in the observed sample consistent with rational choice. This measure (the HM-index) is calculated as the maximal subset of observations consistent with some revealed preference axiom. The method has the additional advantage that researchers can restrict further analysis of the data set to this maximal subset. This paper proposes two simple and fast methods to calculate the HM-index.

Choi, Fisman, Gale, and Kariv (2007) tested whether the choices of 93 subjects over 50 decision rounds were consistent with utility maximisation. Since most subjects were inconsistent, they computed the HM-index to obtain a measure of the degree of inconsistency. However, in doing so, they report that the HM-index is 'computationally intensive even for moderately large data sets' . We apply our procedures to their data and find them to be very fast: the first procedure found a solution for every subject in at most 0.4 seconds, while the second found a solution in at most 3.3 seconds for every subject. Given the efficiency of our algorithms, researchers can use the procedures to run extensive Monte-Carlo simulations to approximate the test power based on the HM-index.

Our first method is a simple combinatorial algorithm based on Gross and Kaiser (1996). It is only applicable for two-dimensional data sets, but because experimental data sets are often two-dimensional, the method is useful for many purposes. As described above, it runs very fast and to our knowledge is the first efficient algorithm that does not require optimisation software. The second method is based on solving a mixed integer linear programming (MIP) problem. This method is not restricted by the dimension of the data set. As such, it is more general but slower than the first procedure. Implementations of the methods for Matlab® and Wolfram Mathematica® are available as supplementary material upon request.

Dean and Martin (2013) recently proposed a new measure of how close choice data is to satisfy utility maximisation. The algorithm used to implement this measure can also be used to calculate the HM-index, and like our second method it consists of solving an MIP problem. Thus, the two problems are similar in computational complexity. But while our MIP problem is deduced directly from the definition of the HM-index, Dean and Martin (2013)'s MIP problem is based on solving the so called 'minimum set covering problem' which is shown to be equivalent to calculating the HM-index. In this respect, it is important to note the simplicity of our first algorithm which does not require using any optimisation packages.

## 2 preliminaries

The commodity space is R L + and the price space is R L ++ , where L ≥ 2 is the number of different commodities. A budget set is defined as B i = B ( p i , w i ) = { x ∈ R L + ∶ p i x i ≤ 1 } , where p i = ( p i 1 , . . . , p i L ) ′ ∈ R L ++ is the price vector and income is normalised to 1. We assume that p i x i = 1; the only observables of the model are N budgets and the corresponding consumer demand. As price vectors characterise budgets, the entire set of N observations is denoted Ω = {( x i , p i )} N i = 1 .

The bundle x i is directly revealed preferred to a bundle x , written x i R 0 x , if p i x i ≥ p i x ; it is strictly directly revealed preferred to x , written x i P 0 x , if p i x i &gt; p i x ; it is revealed preferred to x , written x i R x , if R is the transitive closure of R 0 , that is, if there exists a sequence x j , . . ., x k , such that x i R 0 x j R 0 . . . x k R 0 x . The bundle x i is strictly revealed preferred to x , written x i P x , if x i R x j P 0 x k R x for some j , k = 1, . . . , N .

Axiom (Samuelson 1938) A set of observations Ω satisfies the Weak Axiom of Revealed Preference (Warp) if for all i , j = 1, . . . , N, it holds that [ not x i R 0 x j ] whenever x j R x i and x i ≠ x j .

Axiom (Varian 1982) A set of observations Ω satisfies the Generalised Axiom of Revealed Preference (Garp) if for all i , j = 1, . . . , N, it holds that [ not x i P 0 x j ] whenever x j R x i .

Axiom (Banerjee and Murphy 2006) A set of observations Ω satisfies the Weak Garp (WGarp) if for all i , j = 1, . . . , N, it holds that [ not x i P 0 x j ] whenever x j R 0 x i .

Varian (1982) showed that Garp is a necessary and sufficient condition for the existence of a continuous, monotonic, and concave utility function to rationalise Ω. Banerjee and Murphy (2006) showed that for L = 2, Garp and WGarp are equivalent.

Houtman and Maks (1985) introduced the HM-index to measure the degree of inconsistency of Garp. To formally define the HM-index, let v = ( v 1 , . . . , v N ) be a vector of binary variables (i.e., v i ∈ { 0, 1 } for all i = 1, . . . , N ), and define the relation R 0 ( v ) as x i R 0 ( v i ) x j if v i p i x i ≥ p i x , and let R ( v ) be the transitive closure of R 0 ( v ) ; furthermore, let P 0 ( v i ) if v i p i x i &gt; p i x .

Axiom A set of observations Ω satisfies Garp ( v ) for some v ∈ { 0, 1 } N if for all i , j = 1, . . . , N, it holds that [ not x i P 0 ( v i ) x j ] whenever x j R ( v j ) x i .

Definition The Houtman-Maks (HM) index is the maximal fraction of non-zero elements in the binary vector v such that Garp ( v ) holds.

Thus, the HM-index is the solution to the problem

<!-- formula-not-decoded -->

## 3 the two-dimensional case

As Houtman and Maks (1985), Gross and Kaiser (1996) took a graph-theoretic approach. Every observation is interpreted as a node of a graph. In their definition, if observations i and j form a violation of W arp, then the nodes for i and j are adjacent. The degree of a node i , degr ( i ) , is the number of nodes to which it is adjacent. Define A i as the set of nodes adjacent to node i , and 1 A i as the set of nodes which are adjacent to i with degree 1.

The algorithm consists of two parts. First, whenever degr ( i ) = max j ∈{ 1,..., N } degr ( j ) and degr ( k ) &lt; degr ( i ) for all k ∈ A i , remove i . Repeat this step until no index is removed anymore. Second, whenever degr ( i ) = degr ( h ) = max j ∈{ 1,..., N } degr ( j ) and h ∈ A i , then (1) if 1 A i ≠ ∅ , remove i , (2) if 1 A h ≠ ∅ , remove h , (3) if 1 A i = 1 A h = ∅ , remove either i or h . Again, repeat this step until no index is removed anymore. All nodes not removed in this process belong to the set of indices consistent with W arp. Gross and Kaiser (1996) point out that there is a special case in which the algorithm will fail to provide a maximal subset. However, they argue that this case is extremely rare, and in any case, the algorithm provides a lower bound.

The algorithm is very efficient and easy to implement and therefore suitable for practical purposes. It can easily be adapted for WGarp by simply redefining adjacency. 1 If two nodes i and j are defined as adjacent

1 This has already been noted by Heufer (2014) who did not go into the details of the approach.

whenever i and j form a violation of WGarp, then the same algorithm will provide the set of indices consistent with WGarp. With Banerjee and Murphy's (2006) result, we then have a computationally efficient method to compute the maximal subset of indices which are consistent with Garp in the two-dimensional case.

The same method can also be applied to compute a homothetic HM-index for homotheticity in the two-dimensional case as Heufer (2013) showed that a pairwise version of Varian's (1983) homothetic axiom is sufficient in that case.

## 4 a mixed integer linear programming approach

Adirect way to calculate the HM-index in the multi-dimensional setting (i.e., L &gt; 2) is to numerically solve (1). However, this problem may become difficult to solve because of the complexity in implementing the Garp ( v ) -constraint. Instead we suggest to reformulate the problem as a simple mixed integer programming (MIP) problem by replacing Garp ( v ) with an equivalent condition.

Theorem 1 For any v ∈ { 0, 1 } N , the following conditions are equivalent:

1. the set of observations Ω satisfies Garp ( v ) ;
2. there exist numbers U i ∈ [ 0, 1 ) and ψ i j ∈ { 0, 1 } such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all i , j = 1, . . . , N, where A i is a number greater than p i x i .

This theorem states that Garp ( v ) is equivalent to the set of linear inequalities (2a)-(2d). 2 We suggest to replace Garp ( v ) with the linear inequalities (2a)-(2d) in (1) and calculate the HM-index by solving the following mixed-integer programming problem:

<!-- formula-not-decoded -->

This problem gives an exact and global solution (because every local solution to a MIP problem is a global solution), and there exist efficient algorithms for solving such MIP problems in practice (branch and bound, cutting plane, etc.).

## 5 empirical application and concluding remarks

We applied our methods to data from Choi et al. (2007). This data consists of portfolio choice allocations in a two-dimensional setting (i.e., L = 2) from 93 experimental subjects over 50 decision rounds (i.e., N = 50). Choi et al. (2007) reported the HM-index for all but six subjects which they were unable to find an optimal solution for. We calculated the HM-index for every subject including the six unreported subjects (detailed

2 The proof of Theorem 1 follows (with some simple modifications) from Cherchye, Demuynck, De Rock, and Hjertstrand (2014) who prove a similar theorem in the context of weakly separable utility maximisation.

results for each subject is available as supplementary material upon request). We found the method based on the Gross-Kaiser algorithm to be very fast: it found a solution for every subject in at most 0.4 seconds. 3 Solving the MIP problem (3) is more involved and took a maximum of 3.3 seconds for every subject, which, in our view, can be considered fast enough given the relatively large N . 4

This paper introduced two simple and efficient algorithms for computing the Houtman-Maks-index. The first algorithm is applicable for the two-dimensional setting and does not require any optimisation software. The second algorithm is based on solving a mixed-integer programming problem and can be applied to any dimensional setting. Both of these algorithms can be modified to calculate the HM-index for other revealed preference axioms, such as those for homotheticity.

## references

- Banerjee, S. and Murphy, J. H. (2006): ' A Simplified Test for Preference Rationality of Two-Commodity Choice,' Experimental Economics 9(9):67-75.
- Cherchye, L., Demuynck, T., De Rock, B., and Hjertstrand, P . (2014): 'Revealed Preference Tests for Weak Separability: An Integer Programming Approach,' Journal of Econometrics forthcoming.
- Choi, S., Fisman, R., Gale, D., and Kariv, S. (2007): 'Consistency and Heterogeneity of Individual Behavior under Uncertainty,' American Economic Review 97(5):1921-1938.
- Dean, M. and Martin, D. (2013): 'Measuring Rationality with the Minimum Cost of Revealed Preference Violations, ' Working paper.
- Gross, J. and Kaiser, D. (1996): 'Two Simple Algorithms for Generating a Subset of Data Consistent With WARP and Other Binary Relations,' Journal of Business and Economic Statistics 14(2):251-255.
- Heufer, J. (2013): 'Testing Revealed Preferences for Homotheticity with Two-Good Experiments, ' Experimental Economics 16(1):114-124.
- (2014): 'Nonparametric Comparative Revealed Risk Aversion,' Journal of Economic Theory 153:569616.
- Houtman, M.andMaks, J. (1985): 'Determining all Maximal Data Subsets Consistent with Revealed Preference, ' Kwantitatieve Methoden 19:89-104.
- Samuelson, P. A. (1938): ' A Note on the Pure Theory of Consumer's Behavior, ' Economica 5(17):61-71.
- Varian, H. R. (1982): 'The Nonparametric Approach to Demand Analysis,' Econometrica 50(4):945-972.
- (1983): 'Non-parametric Tests of Consumer Behaviour,' Review of Economic Studies 50(1):99-110.

3 The mean running time over all subjects was 0.09 seconds with a variance of 0.004 seconds.

4 The mean running time for the MIP solution over all subjects was 1.42 seconds with a variance of 0.26 seconds.