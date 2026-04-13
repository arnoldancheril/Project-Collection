# Havyn React Native App Documentation

## Project Overview

Havyn is a cross-platform mobile application developed with React Native and Expo, designed to help users find compatible roommates. The app features a user-friendly interface with swipe functionality (similar to dating apps), detailed profile views, property listings, and in-app messaging. Goal is to convert the RoommateSwipe swift ios application into an React Expo app in the folder HavynReact.

## Development Progress

### Current Status

The Havyn application is currently under development with the following components:


I am creating an application where people can go find roommates in the city of Chicago. I need help creating the UI design for the application. This is a rendition of the logo I had in mind, can you help me design the loading screen for the phone application. 

1. **Authentication**
   - Login and registration with Firebase Auth
   - Forgot password functionality
   - Email verification
   - Welcome screen for new users

2. **Modern UI Components**
   - SwipeCard with detailed profile and property information
   - DetailedProfileScreen with tabbed interface for About, Preferences, and Space details
   - PropertyDetailsScreen with card-style information layout
   - MatchesScreen with tabs for Matches and Group Chats
   - Advanced Filter Modal for refining roommate search
   - Consistent color scheme and styling across all screens

3. **Map Integration**
   - Interactive map view of properties with custom controls
   - Property pins with detailed information
   - Property details preview and navigation
   - Interactive 3D globe view centered on Chicago
   - Toggle between map styles (standard, 3D, satellite)
   - Floating map control buttons for better UX
   - Improved property card previews on marker selection

4. **User Profile**
   - Profile editing (currently placeholder)
   - Lifestyle questions
   - Preferences management

5. **Profile Context System**
   - Context provider for managing liked and matched profiles
   - Persistent storage with AsyncStorage
   - Functions for liking, disliking, and managing profiles
   - Automatic matching simulation with 30% probability
   - Serialization of profile data for navigation state

6. **Swipe Interface**
   - Advanced gesture-based swipe system with smooth animations
   - Visual feedback during swipes (LIKE/PASS indicators)
   - Improved card design with better information hierarchy
   - Modern card animations with rotation and scaling effects
   - Filter system with Location, Budget, Gender, Age, and Lifestyle preferences
   - "More Info" button for detailed profile view

7. **Bottom Navigation**
   - Modern tab bar with visual indicators for active tab
   - Optimized for different devices (including iPhone X-style safe areas)
   - Custom tab bar icons with active/inactive states
   - Improved visual feedback and spacing
   - Consistent navigation structure across all tabs

8. **Profile Management**
   - LikedScreen displaying profiles from context
   - MatchesScreen with tabs for matches and group chats
   - Enhanced DetailedProfileScreen with image gallery and action buttons
   - Added persistence for liked and matched profiles

### Implemented Features

1. **Authentication**
   - Login and registration with Firebase Auth
   - Forgot password functionality
   - Email verification
   - Welcome screen for new users

2. **Map Integration**
   - Interactive map view of properties with multiple view modes
   - Property pins with detailed information
   - Property details preview and navigation
   - 3D globe view for Chicago
   - Fallback UI for map loading issues

3. **User Profile**
   - Profile viewing and editing
   - Matching algorithm based on preferences
   - Swipe-based interface for browsing profiles

4. **Gesture Interface**
   - Advanced swipe gestures using React Native Gesture Handler
   - Smooth animations with React Native Reanimated
   - Visual feedback for user actions
   - Responsive design across different screen sizes

5. **Chat Functionality**
   - Direct messaging between matched users
   - Group chats for property discussions
   - Notification system for new messages

6. **Filter System**
   - Modern, bottom sheet modal interface
   - Budget range selection with sliders
   - Neighborhood selection with chip selectors
   - Property type filtering
   - Gender preference options
   - Age range selection
   - Lifestyle preferences (smoking, pets, drinking, cleanliness)
   - Reset and Apply filter actions

### To Be Implemented

1. **Matching Algorithm**
   - Improved roommate compatibility matching
   - Filtering based on preferences

2. **Notifications**
   - Push notifications for matches and messages
   - In-app notifications

3. **Property Manager Dashboard**
   - Analytics for property listings
   - Interested users management

4. **Payment Integration**
   - For application fees or security deposits

## Known Issues and Limitations

1. **Development Authentication**
   - Firebase authentication is currently bypassed for development purposes
   - Any username/password combination will work in the login screen
   - Use **admin/password** credentials for special admin privileges (property manager mode)
   - This is a temporary solution until Firebase is properly configured

2. **Navigation Structure**
   - All screen navigation has been properly configured and fixed
   - GroupChatDetail and CreateGroupChat screens have been added to the navigation structure
   - Previous navigation errors have been resolved

3. **Firebase Configuration**
   - Firebase authentication requires valid API credentials
   - Currently displays API key errors but allows navigation exploration

4. **Asset Loading**
   - Some placeholder images are used where final assets will be needed
   - Profile images are temporarily using placeholders

5. **Dependency Compatibility**
   - Package versions have been updated for compatibility with Expo SDK
   - React Native Reanimated and Gesture Handler configurations have been optimized
   - All critical package incompatibilities have been resolved

6. **Map Functionality**
   - MapScreen provides a fallback list view when react-native-maps is unavailable
   - Location permissions might need manual configuration on some devices

## Technologies and Libraries

- **React Native**: Core framework for mobile development
- **Expo**: Development toolkit for React Native
- **Firebase**: Authentication, database, and storage
- **React Navigation**: Navigation management with bottom tabs and stack navigators
- **React Native Paper**: UI components and theming
- **Expo Vector Icons**: Icon set integration
- **React Native Gesture Handler**: Advanced gesture recognition for swipe interactions
- **React Native Reanimated**: High-performance animations and transitions
- **@react-native-async-storage/async-storage**: Persistent storage for profile data
- **React Native Maps**: Interactive map integration for property listings
- **@react-native-community/slider**: Range slider component for filters
- **Expo Linear Gradient**: Gradient effects for cards and backgrounds

## Project Structure

```
HavynReact/
├── assets/                    # Static assets like images and icons
│   ├── person-placeholder.jpg # Placeholder profile image
│   ├── adaptive-icon.png      # Adaptive icon for Android
│   ├── favicon.png            # Web favicon
│   ├── icon.png               # App icon
│   ├── logo.png               # App logo
│   └── splash-icon.png        # Splash screen image
│
├── ProfileImages/             # Sample profile images for development
│   ├── profile1/              # Sample profile 1 images
│   ├── profile2/              # Sample profile 2 images
│   ├── profile3/              # Sample profile 3 images
│   ├── profile4/              # Sample profile 4 images
│   ├── profile5/              # Sample profile 5 images
│   ├── profile6/              # Sample profile 6 images
│   ├── profile7/              # Sample profile 7 images
│   ├── profile8/              # Sample profile 8 images
│   └── person-placeholder.jpg # Default profile image
│
├── src/
│   ├── components/            # Reusable UI components
│   │   ├── Button.js          # Custom button component
│   │   ├── ChatBubble.js      # Message bubble for chat
│   │   ├── FormInput.js       # Form input with validation
│   │   ├── PropertyCard.js    # Property listing card
│   │   └── SwipeCard.js       # Card for roommate swiping with gesture support
│   │
│   ├── config/                # Configuration files
│   │   └── firebaseConfig.js  # Firebase setup
│   │
│   ├── contexts/              # React context providers
│   │   ├── AuthContext.js     # Authentication state management
│   │   └── ProfileContext.js  # Profile data management (likes/matches)
│   │
│   ├── hooks/                 # Custom React hooks
│   │
│   ├── models/                # Data models
│   │   └── UserProfile.js     # User profile model
│   │
│   ├── navigation/            # Navigation configuration
│   │   ├── AppNavigator.js    # Main navigator
│   │   ├── AuthStack.js       # Auth flow navigation
│   │   ├── RoommateStack.js   # Roommate user navigation with modern tab bar
│   │   └── PropertyManagerStack.js # Property manager navigation
│   │
│   ├── screens/               # Screen components
│   │   ├── auth/              # Authentication screens
│   │   │   ├── LoginScreen.js          # Login form
│   │   │   ├── SignupScreen.js         # Registration form
│   │   │   ├── ForgotPasswordScreen.js # Password recovery
│   │   │   ├── ProfileTypeScreen.js    # User type selection
│   │   │   ├── SplashScreen.js         # App splash screen
│   │   │   └── BasicInfoScreen.js      # Basic profile info
│   │   │
│   │   ├── messages/          # Messaging screens
│   │   │   ├── ChatScreen.js           # Individual chat conversation
│   │   │   ├── MessagesScreen.js       # List of conversations
│   │   │   ├── GroupChatDetail.js      # Group chat conversation
│   │   │   └── CreateGroupChat.js      # Create new group chat
│   │   │
│   │   ├── profile/           # User profile screens
│   │   │   └── ProfileScreen.js        # User's own profile
│   │   │
│   │   ├── roommate/          # Roommate-finding screens
│   │   │   ├── SwipeScreen.js          # Roommate swiping with gesture controls
│   │   │   ├── DetailedProfileScreen.js # Profile details with sections
│   │   │   ├── PropertyDetailsScreen.js # Property details with card layout
│   │   │   ├── MatchesScreen.js        # Matches with tabbed interface
│   │   │   ├── LikedScreen.js          # Liked profiles
│   │   │   ├── MapScreen.js            # Property map with enhanced controls
│   │   │   └── SettingsScreen.js       # App settings
│   │   │
│   │   ├── propertyManager/   # Property manager screens (in progress)
│   │   │   └── SettingsScreen.js       # Property manager settings
│   │   │
│   │   ├── testing/           # Testing screens
│   │   │   └── TestScreen.js  # Expo test screen
│   │   │
│   │   └── LoadingScreen.js   # Loading screen
│   │
│   ├── services/              # API and service functions
│   │   ├── AuthService.js     # Authentication functions
│   │   ├── UserService.js     # User data operations
│   │   └── PropertyService.js # Property data operations
│   │
│   └── utils/                 # Utility functions
│       ├── helpers.js         # Helper functions
│       ├── sampleProfiles.js  # Sample roommate profiles data
│       └── theme.js           # Theme configuration with modern blue-based color scheme
│
├── .gitignore                 # Git ignore file
├── App.js                     # Main App component with navigation setup
├── babel.config.js            # Babel configuration with Reanimated plugin
├── index.js                   # Entry point
├── app.json                   # Expo configuration
├── package.json               # Project dependencies
├── PROJECT_DOCUMENTATION.md   # This file
└── README.md                  # Project readme
```

## File Descriptions

### Root Files

- **App.js**: Root component that sets up providers and navigation structure with TabNavigator
- **index.js**: Application entry point that registers the main App component
- **babel.config.js**: Babel configuration with presets for Expo and plugins for React Native Reanimated
- **app.json**: Expo configuration file defining app name, version, icons, and platform-specific settings
- **package.json**: NPM package configuration with dependencies, scripts, and metadata
- **README.md**: Project overview, setup instructions, and usage guidelines

### Components

- **Button.js**: Reusable button component with multiple variants (filled, outlined)
- **ChatBubble.js**: Component for rendering messages in the chat interface
- **FormInput.js**: Text input component with integrated validation and error display
- **PropertyCard.js**: Card component for displaying property listing information
- **SwipeCard.js**: Card component for the swipe interface with advanced gesture handling, animations, and visual feedback

### Screens

#### Authentication Screens
- **LoginScreen.js**: User login with email/password
- **SignupScreen.js**: New user registration
- **ForgotPasswordScreen.js**: Password recovery flow
- **ProfileTypeScreen.js**: Selection between roommate seeker, have room, and property manager
- **BasicInfoScreen.js**: Profile creation - basic information
- **SplashScreen.js**: Initial loading screen with logo

#### Roommate Screens
- **SwipeScreen.js**: Main roommate discovery screen with swipe gestures and filter modal
- **DetailedProfileScreen.js**: Detailed view of a user profile with tabs for About, Preferences, and Space details
- **PropertyDetailsScreen.js**: Property information with card-style layout and action buttons
- **MatchesScreen.js**: Matches with tabbed interface for direct matches and group chats
- **LikedScreen.js**: List of liked profiles with action buttons
- **MapScreen.js**: Map view with property markers and multiple view styles
- **SettingsScreen.js**: App settings and account management

#### Message Screens
- **MessagesScreen.js**: List of conversations with matched users
- **ChatScreen.js**: Individual chat conversation interface
- **GroupChatDetail.js**: Group chat conversation with multiple participants
- **CreateGroupChat.js**: Interface for creating new group chats

#### Profile Screens
- **ProfileScreen.js**: User's own profile with editing capability

#### Property Manager Screens
- **SettingsScreen.js**: Settings screen for property managers

#### Testing
- **TestScreen.js**: Expo feature testing screen

### Services and Configuration

- **firebaseConfig.js**: Firebase initialization and configuration
- **AuthService.js**: Authentication-related API functions
- **UserService.js**: User data management functions
- **PropertyService.js**: Property data management functions

### Contexts

- **AuthContext.js**: Authentication state provider and consumer hook
- **ProfileContext.js**: Context provider for managing liked and matched profiles

### Utils

- **helpers.js**: Common utility functions
- **sampleProfiles.js**: Sample roommate profile data with complete information
- **theme.js**: Centralized theme definitions with modern blue-based color scheme, typography, spacing, etc.

## Development Roadmap

### Short-term Tasks (Next Sprint)
1. Fix Firebase configuration with valid API credentials
2. Enable filters to actually filter the roommate profiles
3. Add date picker for move-in date selection in filter modal
4. Continue UI refinements for consistency across all screens
5. Add remaining screens for Property Manager stack

### Mid-term Goals
1. Implement full chat functionality with Firebase
2. Add property manager dashboard
3. Implement profile photo uploading and management
4. Enhance filtering and search options for roommates
5. Add notifications for matches and messages

### Long-term Goals
1. Payment integration for application fees
2. Advanced analytics for property managers
3. Social media integrations
4. Verification systems
5. Expand to web platform

## Running the Application

To run the application:

1. **Important**: Make sure you are in the HavynReact directory. If you are in the root project directory, you must first change to the HavynReact directory:
```bash
cd HavynReact
```

2. Install dependencies:
```bash
npm install
```

3. Start the Expo development server:
```bash
npm start
```

4. Scan the QR code with Expo Go app on your mobile device or run in a simulator/emulator.

5. Alternative commands:
   - `npm run ios` - to open directly in iOS Simulator
   - `npm run android` - to open directly in Android Emulator
   - `npm run web` - to open in web browser

**Note**: If you encounter dependency errors during installation, you may need to update the package versions in package.json to be compatible with your Expo SDK version.

## Troubleshooting

1. **Empty file errors**:
   - If you see "Empty file" errors for assets, check that all required image files have content.
   - You can copy existing images as placeholders: `cp assets/person-placeholder.jpg ProfileImages/profile1/image1.jpg`

2. **Firebase authentication errors**:
   - Current error: "Firebase: Error (auth/api-key-not-valid.-please-pass-a-valid-api-key.)"
   - To fix: Update firebaseConfig.js with valid Firebase project credentials

3. **Screen navigation errors**:
   - Navigation structure has been fixed to ensure consistent navigation between screens
   - The app now correctly handles navigation between tabs and nested screens

4. **React Native Reanimated issues**:
   - If you see "Native part of Reanimated doesn't seem to be initialized" error
   - Make sure babel.config.js includes 'react-native-reanimated/plugin'
   - Restart your bundler with `npm start -- --clear`

5. **UI inconsistencies**:
   - If UI elements don't match the design, ensure the theme provider is properly applied
   - Check that all components are using colors from the theme

## Testing

The application includes a dedicated test screen that can be accessed when in development mode from the login screen. This screen provides functionality to test various Expo features and device capabilities

## Recent Updates

### Navigation Structure and Screens
- Fixed navigation errors for proper screen transitions
- Added missing screens (GroupChatDetail, CreateGroupChat)
- Improved error handling for undefined route parameters
- Added fallback UI elements and placeholder data when data is missing

### Group Chat Implementation
- Created comprehensive GroupChatDetail screen with:
  - Group information header
  - Multi-user chat interface with sender identification
  - Modern messaging UI with bubbles and timestamp formatting
- Implemented CreateGroupChat screen featuring:
  - Form for creating new group chats with name and description
  - Privacy toggle for public/private groups
  - Topical focus selection with multiple options
  - Character count indicators and validation

### Error Handling and Data Integrity
- Enhanced error handling for undefined objects and route parameters
- Added default values and fallbacks for critical components
- Implemented proper null/undefined checks for all data usage
- Fixed image source errors with fallback to placeholder images

### UI Consistency and Polish
- Maintained consistent styling across all screens
- Added visual indicators for user actions
- Improved keyboard handling for input fields
- Ensured proper text styling and readability

### Advanced Filter Modal Implementation
- Added a comprehensive filter modal in SwipeScreen.js with multiple filter categories:
  - Budget range sliders for setting min and max budget
  - Chicago neighborhood selection with chip selectors
  - Property type filters (Apartment, House, Condo, Studio)
  - Gender preference selection with radio buttons
  - Age range sliders for setting min and max age
  - Lifestyle preferences for smoking, pets, drinking, and cleanliness
- Implemented modern UI with bottom sheet modal design
- Added Apply and Reset functionality for filters
- Created visually engaging chip selectors for multi-select options

### Navigation Structure Improvements
- Fixed navigation structure in App.js to ensure consistent navigation between screens
- Added proper screen configuration for all tabs (Discover, Map, Liked, Matches, Profile)
- Ensured DetailedProfile screen is accessible from all relevant tabs
- Fixed header configurations for better visual consistency

### Data Serialization and Error Handling
- Implemented proper serialization of Date objects in profiles for navigation state
- Fixed text rendering issues in SwipeCard.js and other components
- Added proper error handling for profile properties that might be undefined
- Ensured all text is properly wrapped in Text components

### UI and Styling Enhancements
- Added LinearGradient for better text readability on profile cards
- Improved tab bar design with proper spacing and visual feedback
- Enhanced DetailedProfileScreen with tabbed interface for better organization
- Added proper card styling for a more modern look and feel

### Dependency Updates
- Added @react-native-community/slider for range selection in filters
- Added expo-linear-gradient for visual enhancements
- Updated all dependencies to compatible versions for Expo SDK
- Fixed compatibility issues with React Native Gesture Handler and Reanimated 