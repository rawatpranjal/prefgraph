## Choice via AI

Christopher Kops ∗

Maastricht University

Elias Tsakas †

Maastricht University

February 5, 2026

## Abstract

This paper proposes a model of choice via agentic artificial intelligence (AI). A key feature is that the AI may misinterpret a menu before recommending what to choose. A single acyclicity condition guarantees that there is a monotonic interpretation and a strict preference relation that together rationalize the AI's recommendations. Since this preference is in general not unique, there is no safeguard against it misaligning with that of a decision maker. What enables the verification of such AI alignment is interpretations satisfying double monotonicity. Indeed, double monotonicity ensures full identifiability and internal consistency. But, an additional idempotence property is required to guarantee that recommendations are fully rational and remain grounded within the original feasible set.

Keywords: WARP, preferences, AI JEL codes: D01, D91

∗ Department of Quantitative Economics, Maastricht University, P.O. Box 616, 6200 MD, Maastricht, The Netherlands; Homepage: https://sites.google.com/site/janchristopherkops/home ; E-mail: j.kops@maastrichtuniversity.nl

† Department of Microeconomics and Public Economics, Maastricht University, P.O. Box 616, 6200 MD, Maastricht, The Netherlands; Homepage: www.elias-tsakas.com ; E-mail: e.tsakas@maastrichtuniversity.nl

## 1. Introduction

Suppose that a decision maker (DM) seeks advice from an AI Agent before choosing from a menu of available choices. Based on the recommendations that she receives, she is interested in answering two questions. First, does the AI make rational recommendations? Second, are these recommendations aligned with her own preferences?

In this paper, we address these questions in the context of a choice theoretic framework. In particular, we view the AI as a choice function that makes a recommendation for each menu. The only caveat in comparison with the standard framework is that the AI may misinterpret the menu from which the DM is seeking to choose. As a result, it may ignore some of the choices in the menu (similarly to the literature on consideration sets), or it may even end up making a recommendation that does not belong to the menu. For instance, suppose that the DM asks the AI to choose 'the most interesting paper from the Economics literature on political polarization'. However, while the DM has a clear definition of the Economics literature (e.g., the set of Economics journals), the AI has a broader interpretation and thus ends up recommending a paper from the Annual Review of Political Science.

Such misinterpretations are common in AI's that are based on large language models (LLM's), whose interpretation of a context typically depends on the data on which it has been trained. But more importantly, the AI's interpretation of each context is unobservable to the DM. Throughout the paper, we will only impose a weak monotonicity condition on the interpretation operator, according to which basic implications are interpreted correctly, i.e., the AI understands that a paper belonging to the Economics literature also belongs to the literature on Social Sciences, even if it distorts the meaning of 'Economics literature' and 'Social Sciences literature' respectively.

Going back to our first question, the problem boils down to identifying violations of the usual rationality postulates despite the fact that these are possibly confounded with distortions due to potential misinterpretations of the menus. In particular, we ask whether there is a monotonic interpretation operator and a strict preference relation that together rationalize the observed recommendations.

Our first main result establishes that a single condition in the style of a classic acyclicity condition allows us characterizes the set of rationalizable choice functions. This is a major step in that it guarantees that potential mistakes made by the AI can be traced back solely to the misinterpretations of the choice problems, rather than on choice inconsistencies. As such, training the AI with more data so that it learns to interpret the choice environment correctly, paves the way to fully solve the underlying problem.

Yet, preferences may remain only partially identifiable. Even though we can tell whether the

AI's recommendations are rationalizable, we cannot uniquely identify the preference relation, as recommendations are still distorted by the AI's misinterpretations. Interestingly, preference identification does not guarantee correct interpretation of the choice problems. Instead, the two components of our model (viz., the preference relation and the interpretation operator) can be identified jointly from the choice function if the interpretation operator satisfies a double monotonicity condition. According to this condition, not only do we need the AI to correctly interpret logical implications, but moreover whenever the AI thinks that A implies B, this implication does actually hold. In the context of our earlier example, if the AI interprets the Economics literature as a subset of the Social Sciences literature, it is right to do so.

Our second main result then characterizes the set of choice functions that are rationalizable by a strict preference relation together with a doubly monotonic interpretation operator. This is done, by means of standard conditions which rule out binary cycles and choice reversals.

The latter is enough in order to test whether the AI's recommendations are aligned with the DM's preferences. Yet, this does not mean that the AI's recommendations are the ones an agent with correct interpretation of the world would have made. To achieve the latter, we would need to guarantee that the AI's interpretation operator is grounded, i.e., it interprets choice problems correctly.

Our third main result characterizes the choice functions that are consistent with a strict preference relation and a grounded interpretation operator by means of the usual revealed axiom of revealed preference (WARP), thereby making the AI's recommendations rational in the traditional economic sense of the weak axiom of revealed preferences.

While our paper has been framed in the context of recommendations received by an AI agent -which we view as a relevant and timely application- all our results hold verbatim in any setting where the DM receives choice recommendations by an advisor whose interpretation of the world might be distorted and/or the preferences might differ from the ones of the DM.

This paper is related to the broad literature on choice behavior when only a subset of the available alternatives is considered for choice, because of factors such as norms (Baigent and Gaertner, 1996), framing effects (Salant and Rubinstein, 2008), categorization (Manzini and Mariotti, 2012a), clustering (Kops, 2022), cognitive limitations (Eliaz et al., 2011; Masatlioglu et al., 2012; Lleras et al., 2017) or social influence (Cuhadaroglu, 2017; Borah and Kops, 2020). The main difference of our work is that we also allow for consideration of alternatives that are not in the actual choice set.

The behavioral foundation of our choice procedure also shares similarities to the literature modeling choice as the result of (sequentially) maximizing more than one preference relation (Manzini and Mariotti, 2007, 2012b; Cherepanov et al., 2013; Tyson, 2015; Dutta and Horan,

2015; Horan, 2016; Kops, 2018; Armouti-Hansen and Kops, 2018) or maximizing over tuples of alternative and opportunity costs (Armouti-Hansen and Kops, 2024; Manzini et al., 2025). At a broader level, this paper also relates to applications of choice theoretic models to competitive markets (Eliaz and Spiegler, 2011), finance (Stango and Zinman, 2014) and medical decisionmaking (Katz and George, 2019).

Our work also ties well with some earlier work on AI. Since we assume an AI agent rather than a human being as the central decision-maker, our model has to incorporate how LLMs transform (Vaswani et al., 2017) the data they are fed. An immediate consequence lies in the so-called alignment problem of whether or not the AI's preferences clash with the preferences of a human decision-makers. In this regard, Kim et al. (2024) check whether LLMs are able to learn preferences and provide recommendations when it comes to choice under risk. Caplin et al. (2025) test how well machine learning predictions align with cognitive economic model.

The paper is organized as follows. Section 2 lays out the setup. Section 3 defines the model of choice by an agentic AI. Section 4 provides characterization and identification. Section 5 shows how to extend the baseline model to guarantee internal consistency. Section 6 develops it even further to also ensure groundedness. Section 7 elaborates on some of our modeling assumptions and presents an extension guaranteeing groundedness, but not rationality. Section 8 concludes.

## 2. Setup

Let X be a finite set of alternatives with typical elements denoted by x , y , z etc. P ( X ) denotes the set of non-empty subsets of X with typical elements R , S , T etc. We refer to any such set as a choice problem. A choice function on X is a mapping, c : P ( X ) → X .

## 3. AI Agent

Unlike LLMs which only generate text, an AI agent can translate intent into procedures carried out in the real world. An AI agent can act and make choices. In particular, it can provide recommendations to a DM who seeks its advice.

Modern AI agents have an LLM with a transformer architecture (Vaswani et al., 2017) as their policy core. This architecture enables them to convert text into tokens, but also to further contextualize and interpret them. The AI's interpretation of a context then typically depends on the data on which it has been trained. We assume that training and fine-tuning of the LLM enables the AI to correctly interpret basic logical implications, i.e., the AI understands

the subset relations in P ( X ).

Definition 1. (Interpretation Operator). An interpretation operator is a mapping I : P ( X ) →P ( X ) that satisfies the following basic condition:

<!-- formula-not-decoded -->

Monotonicity is the single key property of the interpretation operator. It ensures that howsoever the AI distorts the meaning of certain choice problems, these distortions still preserve the subset-relationship between the non-empty subsets of X . 1 This is the minimal requirement that training and fine-tuning the AI should achieve.

We can now introduce the choice procedure that we are proposing in this paper. The key feature of this procedure is that the AI agent chooses on behalf of our rational DM. So, the procedure requires that the AI picks the best element from the interpreted choice problem according to a preference relation.

Definition 2. A choice function c : P ( X ) → X is an AI agent's choice (AIC) if there exists a strict preference relation ≻ on X and an interpretation operator I on P ( X ) such that for any choice problem S ∈ P ( X ) ,

<!-- formula-not-decoded -->

Under the AIC, I ( S ) can also be thought of as resulting from a Retrieval-Augmented Generation (RAG) step (Shi et al., 2024), where the AI retrieves a context before generating a recommendation.

## 4. Behavioral Foundation

This section provides a behavioral (i.e., choice-based) characterization of the AIC procedure. It identifies which choice functions are rationalizable by interpretation operator and preference relation.

Let us reiterate here that such a characterization enables any outside observer to verify whether choice data is consistent with the AIC procedure or not. As it turns out, this procedure can be characterized by the following single axiom.

No Shifted Cycles (NSC) 1. For all S 1 , . . . , S n +1 , T 1 , . . . , T n +1 ∈ P ( X )

<!-- formula-not-decoded -->

̸

1 In the context of artificial intelligence (Russell and Norvig, 1995), especially when it comes to heuristic functions, monotonicity is also often referred to as consistency.

̸

NSC is in the style of a classic acyclicity condition such as No Binary Cycles (Manzini et al., 2025). The AI's choice from the T i 's is all about this acyclicity. Indeed, c ( T 1 ) = x 2 , c ( T 2 ) = x 3 , . . . , c ( T n ) = x n +1 , c ( T n +1 ) = x 1 would reveal preferences to be cyclic. 2 But, the definition of AIC guarantees that the AI's preferences are acyclic. That is why NSC restricts the AI's choice to satisfy c ( T n +1 ) = x 1 . The added layer of complexity comes from the AI interpreting choice problems. The AI's choice from the S i 's then establishes that each choice problem T i is interpreted in such a way that the interpreted choice problem I ( T i ) involves x i .

The following result then establishes that NSC provides a behavioral characterization of AIC.

Theorem 3. Let X be a finite set of alternatives. A choice function c on X is an AIC if and only if it satisfies NSC.

Proof.

<!-- formula-not-decoded -->

The AIC is based on two key parameters that enter into the AI's decision-making procedure: the strict preference ranking ≻ and the interpretation operator I . Theorem 3 establishes that in NSC there is a testable condition that can be applied to any given choice data to determine whether this data can be thought of as resulting from an AIC procedure.

Now, suppose we have choice data that is consistent with the AIC logic. The question we address next is about the extent to which the two key parameters of the AIC procedure can be uniquely identified from such data. We first consider the question of identification of the AI's preferences. In contrast to rational choice theory, with theories of bounded rationality like the AIC, there may be multiple possible preferences which can rationalize the same choice data. To check whether the AI ranks x above y , it, thus, seems natural to check whether every possible representation of choices ranks x above y . The following definition is useful to organize the discussion.

Definition 4. Let c be an AIC. We say that x is revealed to be preferred to y by the AI, if for any ( ≻ , I ) that is part of a AIC representation of c , we have x ≻ y .

Checking every possible representation of choices may not be a very practicable method. Fortunately, there is a much simpler way to identify the AI's preferences. It heavily draws on the idea of combining choice reversals and subset relationships which is underlying NSC. Indeed, we define the following binary relation ≻ on X via

<!-- formula-not-decoded -->

2 Proposition 5 and its ensuing paragraph illustrate how NSC can be restated directly in terms of the AI's revealed preferences.

The binary relation ≻ may naturally be incomplete, but never intransitive. What is more, by Lemma 26, ≻ is asymmetric and, by Lemma 27, it is also acyclic. As such, it has a transitive closure which we denote by ≻ ∗ . If x ≻ ∗ y , then x is revealed to be preferred to y . Loosely speaking, this is true because if x ≻ z and z ≻ y , then the transitivity of the underlying strict preference relation allows us to infer that x is indeed revealed to be preferred to y . The question remains whether ≻ ∗ really captures all revealed preferences and, at the same time, not more than that. The next proposition establishes that ≻ ∗ really is the revealed preference.

Proposition 5. Let c be an AIC. Then x is revealed to be 'preferred' to y if and only if x ≻ ∗ y . Proof. Please refer to Appendix A.2.

This result is also illuminating with respect to the characterization of AIC. Indeed, it enables us to restate NSC as the demand that the AI's underlying preferences are acyclic. To see this, note that we can use the revealed preference ≻ ∗ to formally restate NSC in the following way

<!-- formula-not-decoded -->

Proposition 5 furthermore shows that multiple preferences may be consistent with AIC choice data. As such, all of the AI's recommendations may align with the DM's preferences without the underlying preferences truly aligning.

Next, we consider the question of identifying the AI's interpretation operator. Again, to check whether the AI considers x at S for its choice, it seems natural to check whether every possible AIC-representation of the AI's choices specifies that x receives the AI's consideration at S . In a similar way as before, the following definition is useful to organize the discussion.

Definition 6. Let c be a CSC. We say that x is revealed to receive consideration at S ∈ P ( X ) by the AI, if for any ( P, I ) that is part of an AIC representation of c , we have x ∈ I ( S )

It turns out, there also exists a simple way to identify the AI's revealed consideration set. To this end, we define the following consideration set I ∗ on P ( X ) via

<!-- formula-not-decoded -->

Again, the question remains whether I ∗ really captures all revealed consideration and, at the same time, not more than that. The next proposition establishes that I ∗ really is the revealed interpretation operator.

Proposition 7. Let c be an AIC. Then x is revealed to the AI's consideration at T if and only if x ∈ I ∗ ( T )

Example 8. Let X = { x, y, z } denote the set of alternatives and x ≻ y ≻ z the DM's preferences. The AI's recommendations are summarized as follows

| S       | { x, y }   | { y, z }   | { x, z }   | { x, y, z }   |
|---------|------------|------------|------------|---------------|
| I ( S ) | { x }      | { y }      | { x }      | { x, y }      |
| c ( S ) | x          | y          | x          | x             |

Observe that this choice data satisfies NSC. But, the preferences underlying the corresponding AIC are not uniquely identifiable. Indeed, any preference ranking ≻ on X satisfying x ≻ y is a potential candidate. In other words, it is not clear whether the AI ranks z above x and y , in between or below.

## 5. Rational AI Agent Choice

The monotonic interpretation operator of an AIC preserves the subset-relationship between the non-empty subsets of X . But, it does not prevent the AI from interpreting two sets S, T ∈ P ( X ) which are incomparable with respect to set inclusion ( S ⊈ T and S ⊈ T ) into sets sharing a subset-relation ( I ( S ) ⊆ I ( T )).

To address this, we next strengthen the connection between the subset-relationship in P ( X ) and the one in { I ( S ) | S ∈ P ( X ) } . The next definition pins down how one can do so and refines the AIC procedure accordingly.

Definition 9. A choice function c : P ( X ) → X is a rational AI agent's choice (RAIC) if it is an AIC and the corresponding interpretation operator satisfies

```
( T DM ) Double monotonicity For all S, T ∈ P ( X ) : I ( S ) ⊆ I ( T ) ⇔ S ⊆ T
```

Double Monotonicity does not only ensure that the interpretation operator preserves the subset-relationship between the non-empty subsets of X . Rather, it also matches every subsetrelation in { I ( S ) | S ∈ P ( X ) } uniquely with the corresponding subset-relation in X prior to interpretation. In other words, the posets ( P ( X ) , ⊆ ) and ( { I ( S ) | S ∈ P ( X ) } , ⊆ ) are order isomorphic.

Next, we turn to a behavioral characterization of the RAIC procedure. Three axioms characterize the rational version of the AIC procedure.

The first axiom rules out pairwise cycles. It is a standard and well-known condition in choice theory, here given in a slightly restated formulation.

Axiom 1 (No Binary Cycles (NBC)) . For all x, y, z ∈ X :

̸

<!-- formula-not-decoded -->

̸

The formulation of NBC here restricts attention to which choices from binary sets differ and which have to coincide. This way, it avoids taking a stance on how exactly the AI interprets the corresponding choice problems.

The second axiom is a condition akin to Contraction Independence (also known as Property α or Chernoff's Condition). In our context, it rules out menu effects by requiring that if an alternative is chosen from a large set, it must also be chosen from any smaller subset whose interpretation contains it.

Axiom 2 (C-Contraction Independence (CCI)) . For all R,S,T ∈ P ( X ) :

<!-- formula-not-decoded -->

At the heart of CCI is the contraction from the set T to its subset S . As for NSC the added layer of complexity comes again from the AI interpreting choice problems. The AI's choice from the set R then establishes that the choice problem S is interpreted in such a way that the interpreted choice problem I ( S ) involves x .

The third axiom specifies that a rational AI agent notices when two alternatives are different. This manifests in the agent's choice behavior as follows.

Axiom 3 (Noticeable Difference (ND)) . For all x, y ∈ X :

̸

<!-- formula-not-decoded -->

̸

Whenever two alternatives are different, ND requires that the agent's choice from the corresponding interpreted singleton sets respects this difference. In other words, the AI's choice from two singleton sets only coincides when these are the same sets.

The following result then establishes that NBC, CCI and ND together form a behavioral characterization of RAIC.

Theorem 10. Let X be a finite set of alternatives. A choice function c on X is an RAIC if and only if it satisfies NBC, CCI, and ND.

Proof.

<!-- formula-not-decoded -->

Next, we analyze what can be inferred from choice data that is consistent with the RAIC logic. While the same exercise for the AIC procedure in Section 4 only allowed to partially identify the AI's preferences and the interpretation operator, for the RAIC we can always fully identify these two key parameters entering into the AI's decision-making procedure. The following result spells this out formally.

Proposition 11. Let c be an RAIC. Then, we can fully identify the AI's interpretation operator I and its preference relation ≻ via defining, for any T ∈ P ( X ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Proof.

An agent is rational in the traditional economic sense if the agent's choice function satisfies the weak axiom of revealed preferences (WARP). WARP demands that if an alternative x is chosen from some set where y is available, then y is never chosen from any set where x is available. As the behavioral characterization in Theorem 10 and the identification in Proposition 11 establish, there is a sense in which the AI's choice under a interpretation operator satisfying Double Monotonicity is rational. But, this does not mean that the AI's choice is necessarily grounded. That is, under a RAIC, it is still possible that c ( S ) / ∈ S , for some S ∈ P ( X ). The next example illustrates this.

Example 12. Let X = { x, y, z } , x ≻ y ≻ z , and I be defined by

<!-- formula-not-decoded -->

Then, the AI's choice is not always grounded. That is, some of its choices satisfy c ( S ) / ∈ S .

| S           | { x } { y } { z }   |       |       | { x, y }   | { y, z }   | { x, z }   | { x, y, z }   |
|-------------|---------------------|-------|-------|------------|------------|------------|---------------|
| I ( S )     | { y }               | { z } | { x } | { y, z }   | { x, z }   | { x, y }   | { x, y, z }   |
| c ( S )     | y                   | z     | x     | y          | x          | x          | x             |
| c ( S ) ∈ S | ✗                   | ✗     | ✗     | ✓          | ✗          | ✓          | ✓             |

Furthermore, the RAIC produces here choices which satisfy WARP on the level of { I ( S ) | S ∈ P ( X ) } , but not on the level of P ( X ) . In other words, the AI chooses as if it maximizes the preferences ≻ on the level of { I ( S ) | S ∈ P ( X ) } .

## 6. Grounded and Rational AI Agent Choice

This section aims to ground a rational AI agent's choice. One way to do so is to prevent the interpretation operator from running into loops, where the AI keeps interpreting already interpreted choice problems. Formally, such looping can be avoided by requiring that the interpretation operator satisfies idempotence.

Definition 13. A choice function c : P ( X ) → X is an AI agent's choice (GRAIC) if it is an RAIC and the corresponding interpretation operator I satisfies

<!-- formula-not-decoded -->

Next, we turn to a behavioral characterization of the RAIC procedure. True to the aim of this section, standard WARP characterizes GRAIC.

Definition 14. A choice function c : P ( X ) → X satisfies WARP if, for all S, T ∈ P ( X ) :

̸

<!-- formula-not-decoded -->

The next result then provides a behavioral characterization of GRAIC. It shows that choice data is consistent with the GRAIC procedure if and only if it satisfies WARP, i.e., is rational in the traditional economic sense.

Theorem 15. Let X be a finite set of alternatives. A choice function c on X is an GRAIC if and only if it satisfies WARP.

Proof.

Please refer to Appendix A.6.

Since a GRAIC is a RAIC, choice data consistent with the GRAIC logic allows to fully identify the AI's preferences and the interpretation operator. The next result shows this formally.

Proposition 16. Let c be an RAIC. Then, we can fully identify the AI's interpretation operator I and its preference relation ≻ via defining, for any T ∈ P ( X ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Please refer to Appendix A.7.

## 7. Elaborations and Extensions

This section elaborates on the consistency of the interpretation operator for AIC and RAIC. It furthermore provides and analysis and characterization of choice behavior by a grounded but potentially irrational AI agent.

and

## 7.1. Monotonic Interpretations

There are two equivalent axiomatizations of the class of interpretations we consider for the AIC in Section 3. These axiomatizations are based on the following properties:

- ( I ∪ ) Union closure For all S, T ∈ P ( X ) : I ( S ) ∪ I ( T ) ⊆ I ( S ∪ T )
- ( I ∩ ) Intersection closure For all S, T ∈ P ( X ) : I ( S ∩ T ) ⊆ I ( S ) ∩ I ( T )

While Monotonicity preserves the subset relationship under the interpretation, Union and Intersection closure are defined entirely on the level of interpretations. Nevertheless, the following result shows the equivalence between the different axiomatizations of interpretations.

Proposition 17 ( Equivalent axiomatizations ) . The following are equivalent:

- (i) An operator I satisfies Monotonicity ( I M ) .
- (ii) An operator I satisfies Union closure ( I ∪ ) .
- (iii) An operator I satisfies Intersection closure ( T ∩ ) .

Proof. Please refer to Appendix A.8.

There are two ways to read the previous result. First, if ( I ∪ ) or ( I ∩ ) seem more appealing than ( I M ), we can simply replace the latter with one of the former in Definition 1. Second, regardless which of the three properties you treat as an axiom, the other two are going to follow as results.

## 7.2. Doubly Monotonic Interpretations

This section shows that interpretations satisfying Double Monotonicity are logically consistent. That is, the interpretation respects that complements cannot overlap. The following result characterizes interpretation operators that are consistent in a number of different ways.

Theorem 18 ( Logical Consistency ) . The next statements are equivalent:

- (i) The interpretation operator I is consistent, i.e.,

<!-- formula-not-decoded -->

- (ii) The interpretation operator satisfies Double Intersection Closure, i.e., we have:

<!-- formula-not-decoded -->

- The interpretation operator satisfies Double Monotonicity, i.e., we have:
- (iii) For all S, T ∈ P ( X ) : S ⊆ T ⇔ I ( S ) ⊆ I ( T ) .
- The interpretation operator I satisfies Negation-Elimination, i.e., we have:
- (iv) For all S ∈ P ( X ) : I ( S c ) = ( I ( S )) c .

̸

- (v) The interpretation operator I is injective, i.e., we have: For all S, T ∈ P ( X ) : S = T ⇒ I ( S ) = I ( T ) .

̸

- (vi) The interpretation operator I is surjective, i.e., we have: For all T ∈ P ( X ) there exists S ∈ P ( X ) such that I ( S ) = T .
- (vii) The interpretation operator I is a relabelling, i.e., it satisfies:
- (a) There exists an automorphism ρ : X → X such that, for all x ∈ X : I ( { x } ) = { ρ ( x ) }
- (b) It satisfies Double Union Closure, i.e., for all S, T ∈ P ( X ) : I ( S ∪ T ) = I ( S ) ∪ I ( T ) .

Proof. Please refer to Appendix A.9.

The previous result implies that there are different ways one can identify inconsistencies. According to part (ii), interpretation is consistent if and only if the AI agent does not make mistakes with conjunctions. According to part (iii), she is correct with all the logical implications. According to part (iv), she is correct with negations (or taking complements). According to part (v) she does not confuse two different sets for being the same. According to part (vi), she associates every choice problem in P ( X ) with some (not necessarily the same) choice problem in P ( X ).

The previous characterizations suggest that a logically consistent AI agent may still make mistakes in what different choice problems actually entail, but does not make mistakes in how different choice problems logically relate to each other. That is why -as the last part of the theorem indicates- the only misinterpretations that are consistent are those that simply relabel the singleton set in P ( X ). In such cases, all the rules of logic are satisfied.

## 7.3. Grounding Irrational and Rational Interpretations

Instead of first ensuring rationalizability and then grounding the AI's choice, this section starts by grounding the choice first. To do so, it defines a grounded interpretation operator as follows.

Definition 19. (Grounded Interpretation). A grounded interpretation is a mapping I : P ( X ) →P ( X ) that satisfies the following two basic conditions:

<!-- formula-not-decoded -->

( I SIP ) Singleton Idempotence For all x ∈ X : I ( I ( { x } )) = I ( { x } )

Next, we introduce the choice procedure of an AI with grounded interpretation.

Definition 20. A choice function c : P ( X ) → X is a grounded AI agent's choice (GAIC) if there exists a strict preference relation ≻ on X and a grounded interpretation I on P ( X ) such that for any choice problem S ∈ P ( X ) ,

<!-- formula-not-decoded -->

As it turns out, this procedure can be characterized by the following single axiom.

Axiom 4 (Groundedness) . For all S ∈ P ( X ) :

<!-- formula-not-decoded -->

The following result then establishes that Groundedness alone forms a behavioral characterization of GAIC.

Theorem 21. Let X be a finite set of alternatives. A choice function c on X is a GAIC if and only if it satisfies Groundedness.

Proof. Please refer to Appendix A.10.

As in previous sections, we next analyze what can be inferred from choice data that is consistent with the GAIC logic. As it turns out, this procedure only allows to identify the absolute minimum, leaving the preferences completely unidentifiable. The following result spells this out formally.

Proposition 22. Let c be an GAIC. Then, for the grounded distortion I , we can identify that, for any T ∈ P ( X ) , it holds that

<!-- formula-not-decoded -->

whereas there is nothing that can be identified about the preference relation ≻ .

Proof. Please refer to Appendix A.11.

Next, we impose monotonicity as an additional property on the interpretation operator. The reason being that this now ensures that we can fully identify the AI's preferences.

Definition 23. A choice function c : P ( X ) → X is a grounded &amp; monotonic AI agent's choice (GMAIC) if it is an GAIC and the corresponding interpretation I satisfies

<!-- formula-not-decoded -->

Indeed, the following result shows that WARP characterizes monotonic version of the GAIC procedure.

Theorem 24. Let X be a finite set of alternatives. A choice function c on X is a GMAIC if and only if it satisfies WARP.

Proof.

<!-- formula-not-decoded -->

As intended, choice data consistent with the GMAIC logic allows to fully identify the AI's preferences. In addition, also the interpretation is fully identifiable as the next result shows.

Proposition 25. Let c be an GMAIC. Then, we can fully identify the AI's grounded interpretation I and its preference relation ≻ via defining, for any T ∈ P ( X ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Please refer to Appendix A.13.

## 8. Conclusion

This paper proposes and analyzes a model of choice for an agentic AI. We view the AI as a choice function that makes recommendations for each menu. The caveat is that the AI may misinterpret the menu from which the DM is seeking to choose. Our main characterization results show (i) under what conditions the AI is grounded in the sense of only recommending alternatives which are feasible choices, (ii) when we can ensure that the AI's recommendations align with the DM's preferences, and (iii) how we can guarantee that the AI makes recommendations which are internally consistent or rational in the traditional economic sense.

Our framework here suggests several avenues for future research. First, while our model is deterministic, a natural extension is in the direction of stochastic choice. In this case, choice data does not come in the form of a single alternative per choice problem. Instead, an analyst is then able to observe a distribution, representing the shares of a finite number of choices. Second, and

our identification results provide a theoretical benchmark for empirical 'revealed preference' tests on AI agents. Via such tests we can then inspect the latent aspects of these systems, making sure they align with the preferences of the individual on whose behalf the AI agent makes choices.

## A. Proofs

## A.1. Proof of Theorem 3

Proof. Necessity: NSC. For i = 1 , . . . , n , let S i , T i ∈ P ( X ) be such that S i ⊂ T i , c ( S i ) = x i , and c ( T i ) = x i +1 . Further, let S n +1 , T n +1 ∈ P ( X ) be such that c ( S n +1 ) = x n +1 and S n +1 ⊂ T n +1 . For NSC to hold, we have to show that this implies that c ( T n +1 ) = x 1 . To this end, note that, for all i = 1 , . . . , n +1, c ( S i ) = x i implies that x i ∈ I ( S i ). From monotonicity, it then follows that also x i ∈ I ( T i ), for all i = 1 , . . . , n + 1,. Furthermore, for all i = 1 , . . . , n , c ( T i ) = x i +1 implies that x i +1 ∈ I ( T i ). By definition of AIC, choice from T 1 then implies that x 2 ≻ x 1 , choice from T 2 then implies that x 3 ≻ x 2 ,. . . , and choice from T n implies that x n +1 ≻ x n . It follows that x n +1 ≻ x n ≻ · · · ≻ x 1 and, in particular, by transitivity of ≻ , that x n +1 ≻ x 1 . As such, x n +1 ∈ I ( T n +1 ) implies that any AIC satisfies c ( T n +1 ) = x 1

̸

Sufficiency: Suppose that c satisfies NSC. We construct the interpretation operator I : P ( X ) →P ( X ) and the preference relation ≻ explicitly.

First, define, for any T ∈ P ( X ), that

<!-- formula-not-decoded -->

Note that, by this definition I is clearly monotonic.

̸

Next, for any x = y , define a binary relation P ⊆ X × X via

<!-- formula-not-decoded -->

Lemma 26. P on X as defined above is asymmetric.

Proof. Towards a proof by contradiction, suppose that P is not asymmetric, i.e., there exist x 1 , x 2 ∈ X such that x 1 Px 2 and x 2 Px 1 . By definition of P above, this means that there exist two sets S 1 , S 2 ∈ P ( X ) with x 1 , x 2 ∈ I ( S 1 ), x 1 , x 2 ∈ I ( S 2 ), and c ( S 1 ) = x 1 , c ( S 2 ) = x 2 . By definition of I above, it follows that there also exist two sets T 1 , T 2 ∈ P ( X ) such that S 1 ⊂ T 1 , S 2 ⊂ T 2 , and c ( T 1 ) = x 2 , c ( T 2 ) = x 1 . Observe, however, that the fact that c ( S 1 ) = x 1 , c ( T 1 ) = x 2 , S 1 ⊂ T 1 , c ( T 2 ) = x 2 , and S 2 ⊂ T 2 imply that c ( T 2 ) = x 2 . Hence, we have arrived at our desired contradiction.

Lemma 27. P on X as defined above is acyclic.

Proof. Towards a proof by contradiction, suppose that P is not acyclic, i.e., there exist n ∈ N ≥ 1 and x 1 , . . . , x n +1 ∈ X such that x n +1 Px n , . . . , x 2 Px 1 , x 1 Px n +1 . By definition of P above, this means that there exist sets S 1 , . . . , S n +1 ∈ P ( X ) with x 1 , x 2 ∈ I ( S 1 ), . . . , x n , x n +1 ∈ I ( S n ),

̸

̸

x n +1 , x 1 ∈ I ( S n +1 ), and c ( S i ) = x i , for all i = 1 , . . . , n +1. By definition of I above, it follows that there also exist sets T 1 , . . . , T n +1 ∈ P ( X ) such that S i ⊂ T i , for all i = 1 , . . . , n +1, and c ( T 1 ) = x 2 , . . . , c ( T n ) = x n +1 , c ( T n +1 ) = x 1 . This is a clear violation of NSC. Hence, we have arrived at our desired contradiction.

̸

We now verify that ( I, P ) represent c on X as an AIC. To that end, pick any T ∈ P ( X ). Let c ( T ) = x . By our definition of I above, it follows that x ∈ I ( T ). Next, we show that there is no alternative y ∈ I ( T ) with yPx . Suppose to the contrary that there is such a y ∈ I ( T ). By our definition of I above, it follows that there is an S ∈ P ( X ) such that S ⊂ T and c ( S ) = y . On the other hand, yPx implies that there is S ′ , T ′ ∈ P ( X ) such that S ′ ⊂ T ′ and c ( S ′ ) = x , c ( T ′ ) = y . But then, c ( S ′ ) = x , c ( T ′ ) = y , S ′ ⊂ T ′ , and c ( S ) = y . So, NSC implies that c ( T ) = x , for all T ∈ P ( X ) with S ⊂ T , and we have arrived at our desired contradiction. It follows that ¬ [ zPx ], for all z ∈ I ( T ). Furthermore, choice from T is decisive, since, by our definition of P above, it holds that xPz , for all z ∈ I ( T ), z = x . This establishes our desired conclusion.

## A.2. Proof of Proposition 5

Proof. Necessity: Take any pair of x and y without x ≻ ∗ y . Then there exists a strict preference ≻ ′ including ≻ ∗ such that y ≻ ′ x . By the proof of Theorem 3, there exists a interpretation operator I such that ( I, ≻ ′ ) represents c . Since y ≻ ′ x , by definition, x cannot be revealed to be preferred to y .

Sufficiency: We have already shown in Section 4 that if x ≻ y , then x is revealed to be preferred to y . Now, consider the case when x ≻ ∗ y . Since ≻ ∗ is defined as the transitive closure of ≻ , this implies that there exists a sequence ( z m ) m = 1 M in X such that x ≻ z 1 , z 1 ≻ z 2 , . . . , z M ≻ y . In this case, we know that for any ≻ ∗ that is part of the representation, ≻⊆≻ ∗ and, hence x ≻ ∗ z 1 , z 1 ≻ ∗ z 2 , . . . , z M ≻ ∗ y . Further, since ≻ ∗ is transitive it follows that x ≻ ∗ y and, thus, x is revealed to be preferred to y

## A.3. Proof of Proposition 7

Proof. Necessity: Take any pair of S ∈ P ( X ) and any x ∈ I ( S ) with x / ∈ I ∗ ( S ). By the proof of Theorem 3, there exists a interpretation operator I with x / ∈ I ( S ) and a ranking P such that ( P, I ) represents c . Since x / ∈ I ( S ), by definition, x cannot be revealed to receive consideration at S .

̸

Sufficiency: Let x ∈ I ∗ ( T ). Then, c ( S ) = x , for some S ∈ P ( X ) with S ⊂ T . Clearly, c ( S ) = x implies that x ∈ I ( S ). Monotonicity implies that also x ∈ I ( T ). Then, it follows from the definition of the interpretation operator that x ∈ I ( T ) for any I that is part of a AIC representation of these choices.

## A.4. Proof of Theorem 10

̸

Proof. Necessity: First, note that Double Monotonicity implies that, for all S ∈ P ( X ), it holds that | T | = | I ( T ) | . To see this, suppose otherwise. That is, there exists T ∈ P ( X ) such that | T | = | I ( T ) | . Note that then monotonicity implies that there exist R,S ∈ P ( X ) satisfying either R ⊊ S ⊆ T , or, T ⊊ R ⊆ S such that | I ( R ) | = | I ( S ) | . Since R ⊂ S , monotonicity implies that I ( R ) = I ( S ). But, then R ⊊ S and I ( R ) ⊆ I ( S ) together violate Double Monotonicity. This establishes that Double Monotonicity implies that, for all S ∈ P ( X ), it holds that | S | = | I ( S ) | .

̸

̸

NBC. Let x, y, z ∈ X be such that c ( { x, y } ) = c ( { x, z } ) = c ( { y, z } ). For NBC to hold, we have to show that then c ( { x, y } ) = c ( { y, z } ). To see this, first observe that, by the argument laid out above, Double Monotonicity implies that | I ( { x, y } ) | = | I ( { y, z } ) | = | I ( { x, z } ) | = 2. Next, observe that { x, y } and { y, z } are not subsets of one another, so Double Monotonicity implies that I ( { x, y } ) and I ( { y, z }} are also not subsets of one another. But, then monotonicity implies that I ( { x, y } ) , I ( { y, z } ) , andI ( { x, z } ) are all three different 2-element subsets of I ( { x, y, z } ). Hence, c ( { x, y } ) = c ( { x, z } ) = c ( { y, z } ) implies that c ( { x, y } ) = c ( { y, z } ).

̸

̸

CCI. Let R,S,T ∈ P ( X ) be such that R ⊂ S ⊂ T and c ( R ) = c ( T ) = x . For CCI to hold, we have to show that then c ( S ) = x . First, note that c ( T ) = x implies that x ≻ y , for all y ∈ I ( T ) \ { x } . Next, since R ⊂ T , Double Monotonicity implies that I ( R ) ⊂ I ( T ). As such, also x ≻ y , for all y ∈ I ( R ) \ { x } . Since c ( R ) = x , we have x ∈ I ( R ) and, by monotonicity also x ∈ I ( S ). Hence, c ( S ) = x .

̸

ND. Let x, y ∈ X be such that x = y . Observe that since { x } and { y } are not subsets of one another, Double Monotonicity implies that also I ( { x } ) and I ( { y } ) are not subsets of one another. From above, we know that Double Monotonicity implies that, for all S ∈ P ( X ), it holds that | S | = | I ( S ) | . Hence, | I ( { x } ) | = | I ( { y } ) | = 1 and, thus, c ( { x } ) = c ( { y } ).

̸

Sufficiency: Suppose that c satisfies NBC, C-WWARP and Noticeable Difference. We construct the interpretation operator I : P ( X ) →P ( X ) and the preference relation ≻ explicitly.

First, define, for any T ∈ P ( X ), that

<!-- formula-not-decoded -->

̸

Note how this ensures that, for any x ∈ X , the set I ( { x } ) is a singleton. Furthermore, Noticeable Difference implies that, whenever x = y also I ( { x } ) = I ( { y } ).

̸

Next, observe that, for any x ∈ X , Noticeable Difference implies that there exists a unique u x ∈ X such that c ( { u x } ) = x . Therefore, define a binary relation P ⊆ X × X via

<!-- formula-not-decoded -->

Lemma 28. P on X as defined above is complete and transitive.

Proof. Take any x, y ∈ X . By our argument above there exist unique u x , u y ∈ X such that either c ( { u x , u y } ) = x or c ( { u x , u y } ) = y . As such, P is complete. It is straightforward to establish that NBC implies that P is also transitive.

Now, take any T ∈ P ( X ) and let c ( T ) = x . Towards a proof by contradiction, suppose that there exists z ∈ I ( T ) such that zPx . Then, by definition of P above, it follows that c ( { u x , u z } ) = z . But, then c ( { u x } ) = x , c ( { u x , u z } ) = z , and c ( T ) = x together with { u x } ⊂ { u x , u z } ⊂ T violates C-WWARP. Indeed, since P is complete and transitive, have instead xPz , for all z ∈ I ( T ) \ { x } .

## A.5. Proof of Proposition 11

̸

Proof. First, by the proof of Theorem 10, Double Monotonicity implies that, for all S ∈ P ( X ), it holds that | S | = | I ( S ) | . Furthermore, from Noticeable Difference it follows that x = y implies c ( { x } ) = c ( { y } ). As such, defining, for any T ∈ P ( X ),

<!-- formula-not-decoded -->

it holds that | T | = | I ( T ) | . Therefore, I truly is the AI's unique interpretation operator recovered from choice data c .

Observe that the above reasoning implies that, for any x, y ∈ X , there exists u x , u y ∈ X such that I ( { u x } ) = { x } , I ( { u y } ) = { y } , and I ( { u x , u y } ) = { x, y } . As such, c ( { u x , u y } ) reveals whether the AI's preferences satisfy x ≻ y or y ≻ x . Since I ( { u x } ) = { x } and I ( { u y } ) = { y } , defining

<!-- formula-not-decoded -->

truly allows to fully identify the AI's preferences.

## A.6. Proof of Theorem 15

Proof. Necessity: As the proof of Theorem 10 lays out, Double Monotonicity implies that, for all S ∈ P ( X ), it holds that | S | = | I ( S ) | . Furthermore, for any two S, T ∈ P ( X ) with S = T ,

̸

̸

̸

it holds that I ( S ) = I ( T ). As such, Idempotence then implies that I is the identity. That is, for all S ∈ P ( X ), it holds that I ( S ) = S .

̸

WARP. Let S, T ∈ P ( X ) be such that c ( S ) = x , y ∈ S , and x ∈ T . For WARP to hold, we have to show that then c ( T ) = y . From above, it follows that I ( S ) = S and I ( T ) = T . As such, c ( S ) = x and y ∈ S implies that x ≻ y . Furthermore, x ∈ T implies that x ∈ I ( T ). Hence, x ≻ y implies that, by definition of an AIC, c ( T ) = y .

̸

Sufficiency: Suppose that c satisfies WARP. We construct the interpretation operator I : P ( X ) →P ( X ) and the preference relation ≻ explicitly.

First, define, for any T ∈ P ( X ), that I ( T ) = T . Clearly, by this definition, I satisfies Double Monotonicity and Idempotence.

̸

Next, define for all x, y ∈ X with x = y that

<!-- formula-not-decoded -->

Clearly, ≻ as thus defined is a complete and transitive binary relation.

Now, take any T ∈ P ( X ) and let c ( T ) = x . Towards a proof by contradiction, suppose that there exists z ∈ I ( T ) = T such that z ≻ x . Then, by definition of ≻ above, it follows that c ( { x, z } ) = z . But, then c ( { x, z } ) = z , c ( T ) = x , together with z ∈ I ( T ) = T violates WARP. Therefore, we have arrived at our desired contradiction. As such, we can conclude that since ≻ is complete and transitive, we have instead x ≻ z , for all z ∈ I ( T ) \ { x } . Hence, the tuple ( ≻ , I ) rationalizes c .

## A.7. Proof of Proposition 16

Proof. First, by the proof of Theorem 15, Double Monotonicity and Idempotence implies that, for all S ∈ P ( X ), it holds that I ( S ) = S . As such, defining, for any T ∈ P ( X ),

<!-- formula-not-decoded -->

allows to truly recover the AI's unique interpretation operator from choice data c .

Furthermore, defining

<!-- formula-not-decoded -->

truly allows to fully identify the AI's complete and transitive preferences.

## A.8. Proof of Proposition 17

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.9. Proof of Theorem 18

Proof. Before starting with the proof, notice that ( ii ) is equivalent to the weaker condition that states that I ( S ) ⊆ I ( T ) implies S ⊆ T , as the converse already holds by monotonicity (Proposition 17).

̸

( i ) ⇒ ( ii ) : Suppose that I is consistent. We proceed to prove ( ii ) by contraposition, i.e., we will show that S ⊈ T implies I ( S ) ⊈ I ( T ). If S ∩ T = ∅ , then by Proposition 17, we have I ( S ) ∩ I ( T ) = ∅ , and a fortiori I ( S ) ⊈ I ( T ). If S ∩ T = ∅ , by Proposition 17, we have I ( S \ T ) ∩ I ( T ) = ∅ , meaning I ( S \ T ) ⊈ I ( T ). The latter combined with I ( S \ T ) ⊆ I ( S ), implies I ( S ) ⊈ I ( T ), which completes this part of the proof.

( ii ) ⇒ ( iii ) : By combining the condition in ( ii ) with Proposition 17, we obtain

<!-- formula-not-decoded -->

Take a sequence of the states ( x 1 , . . . , x K ), so that each state appears exactly once. By definition

<!-- formula-not-decoded -->

Suppose that at least one of these inclusions is not strict. Then, by Equivalence (1), we have { x 1 , . . . , x k } = { x 1 , . . . , x k +1 } , which is an obvious contradiction. Hence, we get

<!-- formula-not-decoded -->

meaning that | I ( S ) | = | S | for every event S ⊆ X . So, we have | I ( { x } ) | = 1 for every x ∈ X . Furthermore, any two distinct states have a different image under I , by Equivalence (1). Therefore I is an automorphism when applied to the singletons. Finally, observe that it can only be that I satisfies the condition in ( iii ).

( iii ) ⇒ ( iv ) : By the condition in (iii) it follows directly that

<!-- formula-not-decoded -->

̸

( iv ) ⇒ ( v ) : Take any S, T ⊆ X such that S = T . Then, it either holds that S \ T = ∅ , or, that T \ S = ∅ . Observe that WLOG we can assume that it is S \ T = ∅ . Then, there exists x ∈ S such that x / ∈ T . Indeed, for such x it holds that x ∈ T c . By negationelimination, I ( T c ) = ¬ I ( T ). As such, by Monotonicity, our x ∈ S \ T satisfies ∅ ̸ = I ( { x } ) = I ( { x } ∩ T c ) ⊆ I ( { x } ) ∩ I ( ¬ T ) ⊆ I ( T c ) = ¬ I ( T ). It follows, in particular, that I ( { x } ) ⊈ I ( T ). But, since x ∈ S , it holds that ( S ∩ { x } ) = ∅ such that, by Monotonicity, it holds that I ( ∅ ) = I ( x ) = I ( S ∩ { x } ) ⊆ I ( S ) ∩ I ( { x } ) ⊆ I ( S ). In summary, ∅ ̸ = I ( { x } ), I ( { x } ) ⊆ I ( S ), but I ( { x } ) ⊈ I ( T ). Hence, I ( S ) = I ( T ).

̸

( v ) ⇒ ( vi ) : Since I is injective, each element of the codomain P ( X ) is mapped to by at most one element of the domain P ( X ). Since domain and codomain coincide, this implies that each element of the codomain is mapped to by exactly one element of the domain. As such, I is surjective. In other words, for every T ⊆ X , there exists S ⊆ X such that I ( S ) = T .

( vi ) ⇒ ( vii ) : First, we show that there exists an automorphism ρ : X → X such that I ( { x } ) = { ρ ( x ) } , for all x ∈ X . To this end, take any x ∈ X . By I being surjective, there exists S ⊆ X such that I ( S ) = { x } . Observe that, for all y ∈ S , Monotonicity then imply that

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

Since ∅ = I ( { y } ), it follows that I ( { y } ) = { x } . By I being surjective and I 's codomain coinciding with its domain, it furthermore follows that S = { y } . The same argument then implies that there exists an automorphism ρ : X → X such that I ( { x } ) = { ρ ( x ) } , for all x ∈ X .

Second, we show that Double Union Closure holds, i.e., that for every S, T ⊆ X : I ( S ∪ T ) = I ( S ) ∪ I ( T ). By Proposition 17, it holds that I ( S ) ∪ I ( T ) ⊆ I ( S ∪ T ). Therefore, it only remains to show that I ( S ∪ T ) ⊆ I ( S ) ∪ I ( T ).

̸

To show this, suppose to the contrary that there exists S, T ⊆ X such that I ( S ∪ T ) ⊈ I ( S ) ∪ I ( T ), i.e., I ( S ∪ T ) \ ( I ( S ) ∪ I ( T )) = ∅ . Now, take any x ∈ ( I ( S ∪ T ) \ ( I ( S ) ∪ I ( T ))). Since I is surjective, there exists R ⊆ X such that I ( R ) = { x } . In fact, from our automorphism above, we know that R = { y } , for some y ∈ X . That is, I ( { y } ) = { x } . By Monotonicity, y / ∈ S ∪ T . To see this, suppose otherwise, i.e., y ∈ S ∪ T . Observe that WLOG we can assume that y ∈ S . Then, by Monotonicity,

<!-- formula-not-decoded -->

And, thus, x ∈ I ( S ), violating that x ∈ ( I ( s ∪ T ) \ ( I ( S ) ∪ I ( T ))). This shows that y / ∈ S ∪ T .

̸

̸

Therefore, ( { y }∪ S ∪ T ) = S ∪ T . Since I is surjective, this implies that also I ( { y }∪ ( S ∪ T )) = I ( S ∪ T ). By Monotonicity, it furthermore holds that I ( { y } ∪ ( S ∪ T )) ⊋ I ( S ∪ T ). Iterating this further, implies that there exists x ∈ X such that ( { x } ∪ { y, . . . } ∪ S ∪ T ) = X = ( { y, . . . }∪ S ∪ T ) = X \{ x } , but I ( X \{ x } ) = X . Since, by Monotonicity, I ( X ) ⊇ I ( X \{ x } ), this implies that I ( X ) = I ( X \{ x } ) = X which violates the fact that I is surjective. Therefore, I ( S ∪ T ) ⊆ I ( S ) ∪ I ( T ). Hence, I ( S ∪ T ) = I ( S ) ∪ I ( T )

̸

̸

( vii ) ⇒ ( i ) : Take any E ⊆ X and consider I ( S ) ∩ I ( S c ). By Double Union Closure, I ( S ) = ⋃ x ∈ S I ( { x } ) and I ( S c ) = ⋃ x ∈ S c I ( { x } ). Since there exists an automorphism ρ : X → X such that I ( { x } ) = { ρ ( x ) } , for all x ∈ X , it follows that I ( { x } ) = { ρ ( x ) } ̸ = { ρ ( y ) } = I ( { y } ), for any x, y ∈ X with x = y . Hence, I ( S ) ∩ I ( S c ) = ∅ .

## A.10. Proof of Theorem 21

̸

Proof. Necessity: Groundedness. Note that Consistency implies that, for any x ∈ X and all S ⊆ X \ { x } , it holds that I ( { x } ) ∩ I ( S ) = ∅ . Since I maps each non-empty subset of X to a non-empty subset of X , this implies that, for any x ∈ X , the set I ( { x } ) is a singleton set. Furthermore, x = y implies that I ( { x } ) = I ( { y } ). But, then idempotence implies that, for all x ∈ X , it holds that I ( { x } ) = { x } .

̸

As such, it follows from Consistency that, for all S ∈ P ( X ), it holds that I ( S ) ⊆ S . Since c ( S ) ∈ I ( S ), this implies that c ( S ) ∈ S .

Sufficiency: Suppose that c satisfies Groundedness. We construct the grounded interpretation I : P ( X ) →P ( X ) and the preference relation ≻ explicitly.

First, define, for any T ∈ P ( X ), that I ( T ) = c ( T ). Next, let ≻ be any complete and transitive binary relation on X .

Now, take any T ∈ P ( X ) and let c ( T ) = x . Since x ∈ I ( T ) and, in particular, I ( T ) = { x } , maximizing any complete and transitive binary relation complete and transitive binary relation produces c ( T ) = x . Hence, the tuple ( ≻ , I ) rationalizes c .

## A.11. Proof of Proposition 22

Proof. By the proof of Theorem 21, for any T ∈ I ( T ), we can infer that { c ( T ) } ⊆ I ( T ) ⊆ T .

Furthermore, the proof of Theorem 21 establishes that there we cannot infer anything about the preferences ≻ which are part of a GAIC.

## A.12. Proof of Theorem 24

Proof. Necessity: By the poof of Theorem 21, for all x ∈ X , it holds that I ( { x } ) = { x } . Monotonicity then implies that, for any fixed S ∈ P ( X ) and all x ∈ S , it holds that x ∈ I ( S ). By Consistency, we then have I ( S ) = S for any such S . As such GMAIC maximizes a complete and transitive preference relation over non-empty subsets of X , so standard results imply that the corresponding c satisfies WARP.

Sufficiency:Suppose that c satisfies WARP. We construct the grounded interpretation I : P ( X ) →P ( X ) and the preference relation ≻ explicitly.

First, define, for any T ∈ P ( X ), that I ( T ) = T . Next, define

<!-- formula-not-decoded -->

Since c satisfies WARP, ≻ as thus defined is complete and transitive. Furthermore, if c ( T ) = x , then c ( { y } ) = y , for any y ∈ T implies that x ≻ y such that the choice from T is decisive. Hence, the tuple ( ≻ , I ) rationalizes c .

## A.13. Proof of Proposition 25

Proof. First, by the proof of Theorem 24, it follows that, for all S ∈ P ( X ), it holds that I ( S ) = S . As such, defining, for any T ∈ P ( X ),

<!-- formula-not-decoded -->

allows to truly recover the grounded &amp; monotonic interpretation I from choice data c .

Furthermore, defining

<!-- formula-not-decoded -->

truly allows to fully identify the AI's complete and transitive preferences.

## References

- Armouti-Hansen, J. and Kops, C. (2018). This or that? sequential rationalization of indecisive choice behavior. Theory and Decision , 84(4):507-524.
- Armouti-Hansen, J. and Kops, C. (2024). Managing anticipation and reference-dependent choice. Journal of Mathematical Economics , 112:102988.
- Baigent, N. and Gaertner, W. (1996). Never choose the uniquely largest a characterization. Economic Theory , 8(2):239-249.
- Borah, A. and Kops, C. (2020). Choice via social influence. Working paper, University of Heidelberg.
- Caplin, A., Martin, D., and Marx, P. (2025). Modeling machine learning: A cognitive economic approach. Journal of Economic Theory , 224:105970.
- Cherepanov, V., Feddersen, T., and Sandroni, A. (2013). Rationalization. Theoretical Economics , 8(3):775-800.
- Cuhadaroglu, T. (2017). Choosing on influence. Theoretical Economics , 12(2):477-492.
- Dutta, R. and Horan, S. (2015). Inferring rationales from choice: Identification for rational shortlist methods. American Economic Journal: Microeconomics , 7(4):179-201.
- Eliaz, K., Richter, M., and Rubinstein, A. (2011). Choosing the two finalists. Economic Theory , 46(2):211-219.
- Eliaz, K. and Spiegler, R. (2011). Consideration sets and competitive marketing. The Review of Economic Studies , 78(1):235-262.
- Horan, S. (2016). A simple model of two-stage choice. Journal of Economic Theory , 162:372406.
- Katz, J. D. and George, D. T. (2019). Reclaiming magical incantation in graduate medical education. Clinical Rheumatology , 39(3):703-707.
- Kim, J., Kovach, M., Lee, K.-M., Shin, E., and Tzavellas, H. (2024). Learning to be homo economicus: Can an llm learn preferences from choice.
- Kops, C. (2018). (F)lexicographic shortlist method. Economic Theory , 65(1):79-97.

- Kops, C. (2022). Cluster-shortlisted choice. Journal of Mathematical Economics , 102:102726.
- Lleras, J. S., Masatlioglu, Y., Nakajima, D., and Ozbay, E. Y. (2017). When more is less: Limited consideration. Journal of Economic Theory , 170:70-85.
- Manzini, P. and Mariotti, M. (2007). Sequentially Rationalizable Choice. American Economic Review , 97(5):1824 - 1839.
- Manzini, P. and Mariotti, M. (2012a). Categorize then choose: Boundedly rational choice and welfare. Journal of the European Economic Association , 10(5):1141-1165.
- Manzini, P. and Mariotti, M. (2012b). Choice by lexicographic semiorders. Theoretical Economics , 7(1):1-23.
- Manzini, P., Mariotti, M., and ¨ Ulk¨ u, L. (2025). Choice and opportunity costs. The Review of Economic Studies , page rdaf101.
- Masatlioglu, Y., Nakajima, D., and Ozbay, E. Y. (2012). Revealed attention. American Economic Review , 102(5):2183-2205.
- Russell, S. J. and Norvig, P. (1995). Artificial intelligence: A Modern Approach . Upper Saddle River, New Jersey: Prentice Hall.
- Salant, Y. and Rubinstein, A. (2008). (a, f): Choice with frames. The Review of Economic Studies , 75(4):1287-1296.
- Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., and Yih, W.-t. (2024). Replug: Retrieval-augmented black-box language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , pages 8371-8384.
- Stango, V. and Zinman, J. (2014). Limited and varying consumer attention: Evidence from shocks to the salience of bank overdraft fees. The Review of Financial Studies , 27(4):9901030.
- Tyson, C. J. (2015). Satisficing behavior with a secondary criterion. Social Choice and Welfare , 44(3):639-661.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. (2017). Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R., editors, Advances in Neural Information Processing Systems , volume 30.