# File Structure and Responsibilities

This document outlines the modular file breakdown for the Trippi mobile app and what each file/folder does. Every file contains a brief header comment describing its purpose.

## Top-level
- `DEVELOPMENT_SETUP.md` – step-by-step project setup and common tasks
- `mobile/` – Expo app with UI

## mobile/
- `package.json` – scripts and dependencies for the Expo app
- `app.config.ts` – Expo configuration (name, icons, splash, plugins)
- `babel.config.js` – Babel presets/plugins (expo, expo-router)
- `tsconfig.json` – TypeScript configuration and path aliases
- `index.js` – expo-router entrypoint
- `assets/Trippi_logo.png` – app logo asset

### Routes (expo-router)
- `app/_layout.tsx` – root layout and tab navigation
- `app/(tabs)/_layout.tsx` – tabs config (Home, Trips, Plan, Budget, Profile)
- `app/(tabs)/index.tsx` – Home screen with large trip cards and quick actions
- `app/(tabs)/trips.tsx` – Trips list with search/filters and navigation to details + create entry
- `app/trips/create.tsx` – multi-step trip creation wizard
- `app/trip/[id].tsx` – trip detail with tabs (Overview, Itinerary, Members, Expenses)
- `app/plan.tsx` – AI Trip Planner (top), itinerary editor, invite, trip switcher
- `app/budget.tsx` – redesigned budget: category breakdown, per-person, contributions
- `app/profile.tsx` – redesigned profile/settings
- `app/ai.tsx` – dedicated AI Trip Planner sample conversation

### Source
- `src/theme/ThemeProvider.tsx` – theming, spacing, typography, safe-area
- `src/state/TripsStore.tsx` – global trips store (create/select/add member/item/contribution/expense; goal budgets + dates)
- `src/utils/format.ts` – currency and percent formatters
- `src/utils/budget.ts` – budget helpers (category colors, totals)
- `src/components/Logo.tsx` – reusable logo component
- `src/components/Button.tsx` – primary/secondary button
- `src/components/InlineButton.tsx` – compact inline button for list actions
- `src/components/Card.tsx` – surface container
- `src/components/Header.tsx` – simple header
- `src/components/Screen.tsx` – safe-area screen wrapper (optional scroll)
- `src/components/TextField.tsx` – labeled text input
- `src/components/ProgressBar.tsx` – progress indicator
- `src/components/Avatar.tsx` – avatar initial badge
- `src/components/ListItem.tsx` – row list item
- `src/components/Segmented.tsx` – segmented tabs control
- `src/components/FAB.tsx` – floating action button
- `src/components/TripCard.tsx` – hero trip preview card
- `src/components/TripListItem.tsx` – rich trip item row for Trips tab
- `src/components/MemberAvatarsRow.tsx` – overlapping avatars row
- `src/components/CategoryLegend.tsx` – legend for budget categories
- `src/components/TripSwitcher.tsx` – modal-based trip picker
- `src/components/PieChart.tsx` – svg-based pie chart
- `src/components/ExpenseForm.tsx` – keyboard-safe stepped expense form
- `src/components/chat/ChatBubble.tsx` – AI/user chat bubble
- `src/data/sample.ts` – mock trips/members/itinerary/contributions data
- `src/components/trip/ItineraryTimeline.tsx` – grouped timeline
- `src/components/trip/TripOverviewCard.tsx` – overview + budget breakdown
- `src/components/trip/ItineraryList.tsx` – redesigned itinerary list
- `src/components/trip/MemberDetailsModal.tsx` – member details modal
- `src/components/trip/ExpensesSection.tsx` – expenses summary + list
- `src/components/trip/TripProgressRow.tsx` – per-trip goal progress

## Notes
- Use path aliases: `@components/*`, `@theme/*`, `@data/*`
- All UI is styled for a dark, modern look and should be easy to iterate.


