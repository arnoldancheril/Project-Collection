# Havyn App - File Structure and Setup Guide

## Project Overview

Havyn is a mobile application designed to help users find roommates. The app is built using React Native with Expo, and will use Firebase for data storage and authentication.

## Environment Setup

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn
- Expo CLI

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd havyn
```

2. Install dependencies
```bash
npm install
# or
yarn install
```

3. Start the development server
```bash
npm start
# or
yarn start
# or
npx expo start
```

## File Structure

```
havyn/
├── App.tsx                      # Entry point of the application
├── app.json                     # Expo configuration
├── babel.config.js              # Babel configuration
├── package.json                 # Project dependencies
├── tsconfig.json                # TypeScript configuration
├── src/                         # Source code
│   ├── assets/                  # Static assets (images, fonts, etc.)
│   │   ├── images/              # Image assets
│   │   │   ├── LoadingScreendesign.png
│   │   │   ├── Login_Page.png
│   │   │   ├── profile_page.png
│   │   │   └── Logo.png         # Main app logo (USED in LoadingScreen) ✅
│   │   ├── fonts/               # Font assets
│   │   └── icons/               # Icon assets
│   ├── components/              # Reusable components
│   │   ├── auth/                # Authentication-related components
│   │   │   └── LoginForm.tsx    # Login form component
│   │   ├── main/                # Main screen components
│   │   │   ├── RoommateCard.tsx # Swipeable roommate profile card (IMPLEMENTED) ✅
│   │   │   └── SwipeHints.tsx   # Animated swipe direction hints (IMPLEMENTED) ✅
│   │   └── common/              # Common/shared components
│   │       ├── Button.tsx       # Reusable button component
│   │       ├── ButtonPrimary.tsx # Primary gradient button (IMPLEMENTED) ✅
│   │       ├── ButtonOutlined.tsx # Outlined button for Google auth (IMPLEMENTED) ✅
│   │       ├── TextField.tsx    # Form input field with validation (IMPLEMENTED) ✅
│   │       ├── RoommateIllustration.tsx # Handshake illustration for login (IMPLEMENTED) ✅
│   │       ├── HomeHeader.tsx   # Header component for home screen (IMPLEMENTED) ✅
│   │       ├── LoadingDots.tsx  # Animated three-dot loader (IMPLEMENTED) ✅
│   │       ├── HavynLogo.tsx    # Havyn logo with house + magnifier (IMPLEMENTED) ✅
│   │       └── ChicagoSkyline.tsx # Chicago skyline silhouette (IMPLEMENTED) ✅
│   ├── config/                  # Configuration files
│   │   └── firebase.ts          # Firebase configuration (placeholder)
│   ├── hooks/                   # Custom React hooks
│   │   └── useForm.ts           # Form handling hook
│   ├── navigation/              # Navigation configuration
│   │   └── index.tsx            # Navigation setup and types (UPDATED with TabBar styling) ✅
│   ├── screens/                 # Application screens
│   │   ├── auth/                # Authentication screens
│   │   │   ├── LoadingScreen.tsx # Loading/splash screen (IMPLEMENTED) ✅
│   │   │   ├── LoginScreen.tsx  # Login screen (IMPLEMENTED) ✅
│   │   │   └── SignUpScreen.tsx # Sign up screen
│   │   └── main/                # Main application screens
│   │       └── HomeScreen.tsx   # Home screen with swipeable roommate cards (IMPLEMENTED) ✅
│   ├── services/                # API and service integrations
│   ├── styles/                  # Styling utilities
│   │   └── theme.ts             # Theme constants (colors, spacing, etc.) (UPDATED) ✅
│   └── utils/                   # Utility functions
│       └── validation.ts        # Form validation utilities
```

## Screen Structure

### Authentication Flow
1. **LoadingScreen** - Initial splash screen with Havyn logo ✅ **IMPLEMENTED**
2. **LoginScreen** - User login form ✅ **IMPLEMENTED**
3. **SignUpScreen** - New user registration

### Main App Flow
1. **HomeScreen** - Swipeable roommate cards (main feed) ✅ **IMPLEMENTED**
2. **MapScreen** - Map view of available roommates
3. **MatchesScreen** - List of mutual connections
4. **MessagesScreen** - Chat threads
5. **ProfileScreen** - User's profile and settings

## HomeScreen Implementation

The HomeScreen has been fully implemented as the main profile search screen with the following features:

### Visual Elements
- **Gradient Background**: Profile-specific gradient (#E4F2FF → #FFFFFF)
- **Header Component**: Havyn logo with filter icon (HomeHeader.tsx)
- **Swipeable Card Stack**: Tinder-style deck with 85% screen width cards
- **Card Design**: Rounded corners (24dp), hero image, price/bedroom chips overlay
- **Card Footer**: Name, age, location, gender icons with expandable bio section
- **Swipe Hints**: Subtle "Skip" (left) and "Connect" (right) labels with fade animations
- **Bottom Navigation**: 5-tab navigation with pill design and custom styling

### Components Created
- **RoommateCard**: Complete swipeable card with profile information and expandable bio
- **SwipeHints**: Animated hints that respond to drag distance and direction
- **HomeHeader**: Clean header with logo and filter functionality

### Technical Features
- **Pan Responder**: Custom swipe gesture handling with threshold detection
- **Animation System**: Smooth card transitions, hint opacity changes, and scale effects
- **Sample Data**: Three diverse roommate profiles with realistic information
- **Interactive Elements**: Card tap for full profile modal, filter button functionality
- **State Management**: Current index tracking and profile cycling
- **Background Cards**: Stacked card effect with next profile preview

### Swipe Mechanics
- **Gesture Recognition**: 20px minimum movement threshold to initiate swipe
- **Hint Feedback**: Real-time opacity changes based on drag distance (max 30% screen width)
- **Swipe Threshold**: 25% screen width minimum for successful swipe
- **Direction Handling**: Left swipe = Skip, Right swipe = Connect
- **Animation**: 300ms card exit animation with screen-width displacement
- **Reset Logic**: Automatic return to first profile when deck is exhausted

### Bottom Navigation Styling
- **Pill Design**: 72dp height with 32dp border radius
- **Positioned**: Absolute positioning with 16dp margins from screen edges
- **Icons**: Ionicons with filled/outlined states for active/inactive tabs
- **Colors**: Navy primary (#033E6B) for active, secondary gray for inactive
- **Shadow**: Subtle drop shadow for floating effect
- **Safe Area**: Extra bottom padding for iOS devices

## App Flow Implementation

The application now properly flows from LoadingScreen → LoginScreen → HomeScreen:

1. **LoadingScreen**: Shows for 3 seconds with logo animation, then navigates to Login
2. **LoginScreen**: Complete auth interface (currently for display only)
3. **HomeScreen**: Main profile search interface with full swipe functionality

### Navigation Updates
- **Initial Route**: Loading screen is now the entry point
- **Auto-transition**: Loading screen automatically navigates to login after 3 seconds
- **Tab Bar**: Fully styled bottom navigation with proper icons and labels
- **Type Safety**: Complete TypeScript definitions for all navigation parameters

## Dependencies Added
- **react-native-deck-swiper**: For advanced card swiping mechanics (installed with legacy peer deps)
- **@expo/vector-icons**: Vector icons for navigation and UI elements
- **react-native-safe-area-context**: Safe area handling for modern devices
- **@expo-google-fonts/poppins**: Poppins font family for typography
- **expo-linear-gradient**: For gradient backgrounds used across screens
- **react-native-svg**: For custom vector illustrations and icons
- **react-native-gesture-handler**: For advanced gesture handling
- **@react-navigation/native**: Core navigation package
- **@react-navigation/stack**: Stack-based navigation for auth flow
- **@react-navigation/bottom-tabs**: Tab-based navigation for main app screens

## Sample Data Structure

```typescript
interface RoommateProfile {
  id: string;
  name: string;
  age: number;
  location: string;
  price: number;
  bedrooms: string;
  gender: string;
  photo: string;
  bio?: string;
  isPetFriendly?: boolean;
}
```

## Testing the Profile Search Screen

The HomeScreen can be tested with the following features:
- Swipe cards left (skip) or right (connect) to navigate through profiles
- Tap cards to view full profile information in alert modal
- Drag cards partially to see swipe hints fade in/out
- Tap filter icon to see placeholder filter functionality
- Navigate between tabs in bottom navigation
- Experience smooth animations and gesture feedback

## Theme Updates

Added new colors for the profile search screen:
- `primaryProfile`: #033E6B (Profile-specific navy)
- `profileBackgroundGradient`: #E4F2FF → #FFFFFF
- `cardShadow`: rgba(3, 62, 107, 0.08) for 8% opacity shadows
- `white`: #FFFFFF for card backgrounds

## LoginScreen Implementation

The LoginScreen has been fully implemented with the following features:

### Visual Elements
- **Gradient Background**: Sky blue to white gradient (#E7F3FF → #FFFFFF)
- **Header Logo**: Small Havyn icon (48x48px) in top-left corner
- **Hero Section**: "Welcome to Havyn" title with roommate handshake illustration
- **Form Card**: White card with subtle shadow containing all form elements
- **Form Fields**: Email and password with proper validation
- **Primary Button**: Gradient button for login/signup actions
- **Google Auth Button**: Outlined button with Google icon
- **Auth Switcher**: Toggle between login and signup modes

### Components Created
- **TextField**: Reusable input field with validation, error states, and password toggle
- **ButtonPrimary**: Gradient button with loading states and accessibility
- **ButtonOutlined**: Outlined button for secondary actions
- **RoommateIllustration**: Custom SVG illustration of two people shaking hands

### Technical Features
- **Form Validation**: Email format, password requirements, confirm password matching
- **Loading States**: Button shows spinner during form submission
- **Error Handling**: Real-time validation with error messages
- **Accessibility**: Proper labels, roles, and screen reader support
- **Responsive Layout**: Keyboard avoiding view and scroll support
- **Auth Mode Toggle**: Single screen handles both login and signup

### Dependencies Added
- **expo-linear-gradient**: For gradient buttons and backgrounds
- **react-native-svg**: For custom illustrations
- **expo-font**: For Poppins font family (future implementation)

## LoadingScreen Implementation

The LoadingScreen has been fully implemented with the following features:

### Visual Elements
- **Gradient Background**: Sky blue to white gradient (#E7F3FF → #FFFFFF)
- **Havyn Logo**: Actual Logo.png image from assets with pulse animation (replaced custom SVG icon)
- **Typography**: Poppins SemiBold 48pt for "Havyn", Poppins Light 20pt for tagline
- **Chicago Skyline**: Animated silhouette at 20% opacity
- **Loading Dots**: Three-dot wave animation

### Animations
- **Logo Pulse**: 2-second loop with 3% scale increase using React Native's Animated API
- **Skyline Parallax**: 8-second horizontal drift
- **Loading Dots**: 1.2-second wave animation with opacity and Y-translation

### Technical Features
- **Font Loading**: Poppins Light & SemiBold fonts
- **Accessibility**: Screen reader announcements for logo and tagline
- **Auto-transition**: 3-second maximum loading time
- **Responsive Layout**: Centered layout with fixed positions for skyline and loading dots
- **Image Handling**: Proper resizeMode for logo image
- **Performance**: useNativeDriver for hardware-accelerated animations

### UI Updates (Latest)
- **Logo Update**: Implemented actual Logo.png from assets folder (previously was custom SVG)
- **Layout Refinement**: Improved component positioning to match design mockups
- **Responsive Sizing**: Logo size now adapts to screen width for better scaling across devices
- **Animation Timing**: Refined pulse animation timing for subtler effect
- **Accessibility**: Enhanced screen reader support for all elements

## Adding New Screens

1. Create a new screen file in the appropriate folder (auth or main)
```tsx
// src/screens/main/NewScreen.tsx
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const NewScreen = () => {
  return (
    <View style={styles.container}>
      <Text>New Screen</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default NewScreen;
```

2. Add the screen to the navigation
```tsx
// Update src/navigation/index.tsx to include the new screen
import NewScreen from '../screens/main/NewScreen';

// Add the screen to the appropriate navigator
<MainTab.Screen name="NewScreen" component={NewScreen} />
```

## Styling Guide

The app uses a consistent styling approach with a centralized theme file:

- **Colors** - Defined in `src/styles/theme.ts`
- **Spacing** - Consistent spacing values defined in theme
- **Typography** - Font sizes and weights in theme
- **Components** - Reusable styled components in `src/components/common`

## Firebase Setup (To be completed later)

1. Create a Firebase project
2. Enable Authentication and Firestore
3. Update the configuration in `src/config/firebase.ts`
4. Initialize Firebase in the app

## Development Workflow

1. Run the development server: `npm start`
2. Choose a platform to run on:
   - Press `a` for Android emulator
   - Press `i` for iOS simulator
   - Scan QR code with Expo Go app on physical device

## Testing the Application

The complete app flow can now be tested:
1. **Loading Screen**: Shows Havyn branding with smooth animations
   - Observe the logo pulse animation with the official Logo.png
   - Notice the responsive design adapting to different screen sizes
   - Watch the subtle Chicago skyline animation at the bottom
   - See the three-dot loading animation
   - After 3 seconds, it automatically transitions to the Login screen
2. **Login Screen**: Complete authentication interface with form validation
3. **Home Screen**: Swipeable roommate cards with full gesture support
4. **Bottom Navigation**: Tab switching between different app sections

Current test flow: LoadingScreen (3s) → LoginScreen → HomeScreen (swipeable cards)

## Expo Build Instructions

To create a standalone build:

```bash
# Build for Android
expo build:android

# Build for iOS
expo build:ios
```
