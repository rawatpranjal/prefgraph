Identification and Failure Modes
=================================

Revealed preference scores measure the degree to which a sequence of observations is
consistent with utility maximization. The measurement is only valid when the observations
faithfully represent the agent's actual decisions. Several data features commonly found in
retail scanner data, e-commerce logs, and platform recommendation systems can produce
apparent violations that reflect errors in the data rather than genuine inconsistency in
behavior.

The theoretical foundations page enumerates five maintained assumptions. Assumption A5 states
that the analyst observes the exhaustive set of commodities and prices relevant to the
agent's decision. Assumption A4 states that the choices come from a single optimizing agent.
Assumption A1 states that preferences are stable across all observations. Each failure mode
below corresponds to a breach of one of these assumptions, with a plain account of what goes
wrong and what to check.

Stockouts and Unavailability
-----------------------------

Budget revealed preference tests compare what was chosen with what was affordable. If a
consumer appears to choose bundle A over bundle B, the test infers that A is revealed
preferred to B. This inference is valid only if B was genuinely available at the time of
choice. When a good is out of stock, its recorded price is often the shelf or catalog price,
but the consumer could not actually purchase it. The analyst observes what looks like a
budget set that includes the good when in fact the consumer faced a restricted set.

The consequence is a spurious violation. A household that buys no dairy in one week may look
like it switched from dairy-heavy baskets to dairy-free ones, generating an apparent GARP
cycle, when the true explanation is that the preferred product was absent from the shelf.
The check is to audit whether any good has zero quantity across all observations for a user
in a given period and whether that pattern correlates with known stockout events or
distribution gaps. Scanner data sometimes carries an availability or shelf-share field;
using it to filter observations is more reliable than treating zero-quantity as a genuine
choice.

Platform and Recommendation Bias
---------------------------------

When a recommendation engine controls the items displayed to a user, the choice set is
endogenous to the user's own history. A user who is consistently shown product A because the
algorithm predicts she likes it will consistently click product A. If product B is never
shown, the analyst cannot observe whether the user would prefer B over A. The platform's
intervention confounds revealed preferences with algorithmic assignment.

For menu-based analysis, the problem appears when the log records only items that were
presented and clicked, but the engine's selection of what to present was itself informed by
past preferences. In that setting, consistency scores measure the coherence of the
recommendation algorithm as much as the coherence of the user. The check is to determine
whether the menus in the log are algorithmically generated, whether different users were
shown different sets of alternatives, and whether choice set assignment correlates with past
behavior. Datasets where every item in the logged slate was shown to the user before any
click was made are relatively clean on this dimension. Datasets where the platform dynamically
personalizes the display based on the current session are not.

Reconstructed Menus
--------------------

Many e-commerce and clickstream datasets do not directly record the choice set. The analyst
reconstructs a proxy menu from session-level behavior, typically by collecting all items
viewed before a purchase within a time window or session gap. This reconstruction is not the
same as a logged platform menu.

A reconstructed menu conflates consideration with presentation. The user chose which items
to view; the set of viewed items is itself revealed behavior. Testing consistency on a
self-selected consideration set is circular in a way that testing on a platform-logged
slate is not. In the worst case, a user who deliberately browses a small curated set before
purchasing will score as highly consistent regardless of underlying preferences, while a user
who browses broadly before settling on a choice will appear inconsistent simply because more
alternatives appear in the reconstructed menu. The check is to trace whether the menu in the
dataset was logged by the platform at presentation time or reconstructed after the fact from
engagement events.

Bundle Aggregation
-------------------

Budget analysis treats each observation as a single choice: the agent chose this bundle at
these prices in this period. When the underlying data is a transaction log, multiple
purchases made within a short interval are sometimes collapsed into one row. If those
purchases were made on separate trips, under different prevailing prices, or with different
items available, combining them into one bundle creates an observation that no decision-maker
ever faced as a single choice. The budget constraint attached to the aggregated row is a
fiction.

The aggregated bundle may violate a budget constraint that appears tight by construction, or
it may appear to exhaust a budget that is actually the sum of several smaller budgets. Both
outcomes produce violations that are artifacts of the aggregation rather than of the agent's
behavior. The check is to verify that each row in the data corresponds to a single purchase
occasion and that the price vector reflects what the agent actually faced at that moment
rather than an average or imputed price across the aggregation window.

Category Aggregation
---------------------

Revealed preference models assume that the goods in the commodity space are well-defined and
substitutable at the margin. When goods are constructed by aggregating heterogeneous items
into a single category column, the resulting good does not correspond to anything the
consumer actually chose. A category called "dairy" that bundles milk, cheese, yogurt, and
butter at a category-average price is a constructed variable, not an observed quantity.

Within-category substitution can generate apparent cross-period preference reversals. A
consumer who shifts from buying premium cheese to buying budget milk has changed the
within-category composition of their basket, not their preference between categories. Using
a category-level price and a quantity aggregate to test GARP will interpret that
within-category shift as a revealed preference cycle. The severity grows with the
heterogeneity of the items aggregated into each category. The check is to examine the spread
of prices and unit values within each category across periods. If a category's effective
price varies substantially across periods due to composition shifts rather than actual price
changes, category-level analysis will produce misleading consistency scores.

Habit and Repeated Exposure
-----------------------------

Revealed preference tests are silent on why an agent chose what they chose. They test only
whether the pattern of choices could have been generated by a stable utility function. Habit
and familiarity introduce a systematic reason for choices that can interact with consistency
scores in two distinct ways.

A consumer who has not yet encountered a product cannot reveal a preference for it. When a
new product enters the market or is heavily promoted in one period, initial non-purchase
followed by later purchase can look like a GARP violation if the later bundle is cheaper than
what the consumer chose in an earlier period. The actual explanation is that the consumer's
awareness or consideration set changed between periods. The product was technically available
and priced in the data, but awareness was absent.

At the other extreme, a consumer who buys the same basket every week will score high on CCEI
not because of disciplined optimization but because habit produces a degenerate sequence with
no revealed trade-offs. In this case, a high consistency score is uninformative. The check
for habit bias is to examine the fraction of observations that are exact or near-exact repeats
of the preceding period and to assess whether violations cluster in periods when the basket
deviated substantially from the habitual pattern.

Time-Varying Preferences Over Long Panels
------------------------------------------

Assumption A1 requires that the agent's utility function is stable across all observations.
Consumer preferences shift with income, household composition, age, health, and seasonal
patterns. Over short panels of a few weeks or months, this assumption is usually defensible.
Over multi-year panels, it becomes progressively less credible.

The consequence is that GARP violations in long panels may reflect genuine preference change
rather than irrationality. A household that shifts spending from child-focused categories to
retirement-relevant ones as children leave home is not being irrational; it is adapting to a
changed situation. The CCEI for such a household will be depressed even if the household
maximizes utility perfectly at each point in time.

The check is to look for temporal clustering of violations. If violations concentrate at the
boundary between two sub-periods of the panel rather than distributed uniformly, the likely
explanation is preference drift rather than systematic inconsistency. Rolling-window analysis,
where scores are computed on short consecutive windows rather than the full panel, can
separate violations due to noise from those due to structural change. When a panel spans
several years and violation density increases sharply at identifiable life events, the
appropriate response is to segment the panel at those events rather than to treat the full
sequence as a single person's stable preference record.
