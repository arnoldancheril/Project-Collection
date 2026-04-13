# Mobile App File Structure (Updated)

This document lists the key files in the `mobile/` app and their purposes. Each file includes a header comment in code describing its role.

## Routes
- `app/_layout.tsx` – root stack (login + tabs)
- `app/(tabs)/_layout.tsx` – tabs configuration
- `app/(tabs)/index.tsx` – Home (sorted hero trip cards)
- `app/(tabs)/trips.tsx` – Trips list → Trip Detail
- `app/trips/create.tsx` – Trip creation wizard
- `app/trip/[id].tsx` – Trip Detail (Overview, Itinerary with edit/expenses, Timeline, Members, Expenses)
- `app/plan.tsx` – AI Trip Planner (top), Select Trip, Itinerary list, Split Costs
- `app/budget.tsx` – Scrollable budget with per-person, category legend, contributions & balance
- `app/profile.tsx` – Redesigned profile with insights and balances
- `app/ai.tsx` – Dedicated AI Trip Planner sample conversation screen

## State & Data
- `src/state/TripsStore.tsx` – trips store (create/select/add member/item/contribution/expense; edit itinerary; supports goal budget and dates)
- `src/state/AuthContext.tsx` – auth provider; demo-aware (skips Firebase subscription when no env)
- `src/data/sample.ts` – sample trips with itinerary (category, times), contributions, expenses

## Components
- `src/components/Screen.tsx` – safe-area wrapper (supports scrolling)
- `src/components/Card.tsx` – surface card
- `src/components/Button.tsx` – buttons
- `src/components/InlineButton.tsx` – compact inline button for list actions
- `src/components/TextField.tsx` – form fields
- `src/components/Segmented.tsx` – segment control
- `src/components/TripCard.tsx` – hero trip card
- `src/components/FAB.tsx` – floating action button
- `src/components/Avatar.tsx` – avatar
- `src/components/ListItem.tsx` – list row
- `src/components/ProgressBar.tsx` – progress
- `src/components/PieChart.tsx` – budget chart
- `src/components/TripSwitcher.tsx` – modal trip picker
- `src/components/MemberAvatarsRow.tsx` – avatars row
- `src/components/CategoryLegend.tsx` – budget legend
- `src/components/trip/ItineraryTimeline.tsx` – grouped timeline for itinerary
- `src/components/trip/TripOverviewCard.tsx` – overview + budget breakdown card
- `src/components/trip/ItineraryList.tsx` – redesigned itinerary list with compact actions
- `src/components/trip/MemberDetailsModal.tsx` – member details modal with quick actions
- `src/components/trip/ExpensesSection.tsx` – expenses summary + list
- `src/components/trip/TripProgressRow.tsx` – per-trip goal progress row
- `src/components/chat/ChatBubble.tsx` – chat bubble for AI/User messages

## Utils
- `src/utils/format.ts` – currency/percent
- `src/utils/budget.ts` – category colors & totals
- `src/utils/balances.ts` – per-member balances from expenses & contributions

## Firebase
- `src/firebase.ts` – Firebase initialization with AsyncStorage persistence; supports demo mode when `.env` is missing. Exports `auth`, `db`, and `isDemoMode`.

