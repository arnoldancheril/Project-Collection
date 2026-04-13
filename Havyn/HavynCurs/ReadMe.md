# Havyn App Development Updates

## File Structure:

```
HAVYNCURS
├── ProfileImages
└── RoommateSwipe
    ├── RoommateSwipe
    │   ├── Assets.xcassets
    │   ├── Models
    │   │   ├── ApartmentModels.swift
    │   │   ├── Profile.swift
    │   │   └── UserProfile.swift
    │   ├── Preview Content
    │   │   └── Preview Assets.xcassets
    │   ├── Services
    │   │   └── FirebaseService.swift
    │   ├── ViewModels
    │   │   └── RoommateViewModel.swift
    │   ├── Views
    │   │   ├── ApartmentListingSignUpView.swift
    │   │   ├── ApartmentRootTabView.swift
    │   │   ├── ApartmentSettingsView.swift
    │   │   ├── AnalyticsView.swift
    │   │   ├── BasicInfoView.swift
    │   │   ├── ButtonStyles.swift
    │   │   ├── ChatComponents.swift
    │   │   ├── DetailedProfileView.swift
    │   │   ├── DetailedPropertyView.swift
    │   │   ├── FormComponents.swift
    │   │   ├── HaveRoomSignUpView.swift
    │   │   ├── InterestedUsersView.swift
    │   │   ├── LifestyleQuestionsView.swift
    │   │   ├── LikedView.swift
    │   │   ├── ListingViews
    │   │   │   ├── ListingDetailView.swift
    │   │   │   ├── ListingEditView.swift
    │   │   │   └── ListingMapView.swift
    │   │   ├── ListingsView.swift
    │   │   ├── LoadingView.swift
    │   │   ├── LoginView.swift
    │   │   ├── MapPropertyView.swift
    │   │   ├── MapView.swift
    │   │   ├── MatchedProfileView.swift
    │   │   ├── MatchesView.swift
    │   │   ├── PhotoUploadView.swift
    │   │   ├── PreferencesView.swift
    │   │   ├── ProfileDisplayView.swift
    │   │   ├── ProfileEditView.swift
    │   │   ├── ProfileTypeSelectionView.swift
    │   │   ├── ProfileView.swift
    │   │   ├── PropertyPreferencesView.swift
    │   │   ├── RootTabView.swift
    │   │   ├── ScheduleView.swift
    │   │   ├── SignUpView.swift
    │   │   ├── SwipeCardView.swift
    │   │   └── SwipeView.swift
    │   ├── AllCode.txt
    │   ├── ApartmentListingRegistrationData.swift
    │   ├── AppUpdates.md
    │   ├── ContentView.swift
    │   ├── GoogleService-Info.plist
    │   ├── HaveRoomRegistrationData.swift
    │   ├── Info.plist
    │   ├── Item.swift
    │   ├── Notes.txt
    │   ├── RoommateSwipe.entitlements
    │   ├── RoommateSwipeApp.swift
    │   └── UserRegistrationData.swift
    ├── Firebase_Implementation_Summary.md
    ├── Firebase_Setup_Instructions.md
    ├── RoommateSwipe.xcodeproj
    ├── RoommateSwipeTests
    │   └── RoommateSwipeTests.swift
    └── RoommateSwipeUITests
        ├── RoommateSwipeUITests.swift
        └── RoommateSwipeUITestsLaunchTests.swift
```

## Update Log

### March 27, 2024 - Apartment Lister Dashboard Implementation

Added a separate interface for property owners and managers to list and manage their apartments. This implementation includes:

1. New button on the login screen to access the apartment listing dashboard
2. Custom tab bar navigation for property managers with four main sections:
   - Listings: View and manage property listings
   - Interested Users: Track and interact with potential tenants
   - Analytics: View property performance metrics
   - Settings: Configure account and notification preferences
3. UI improvements for a clean, intuitive, and engaging user experience
4. Fixed issues with header spacing and tab appearance in the apartment dashboard

## File Descriptions

### Root Files

- **RoommateSwipeApp.swift**: The main entry point for the application that sets up the SwiftUI app structure and initializes Firebase.
- **ContentView.swift**: The root view that manages navigation between the login screen, loading screen, and the main tab views (both for roommate seekers and apartment listers).
- **Item.swift**: A simple data model for SwiftData integration and persistence.
- **Notes.txt**: Development notes and documentation for the project.
- **AppUpdates.md**: This file, which documents app structure and major updates.
- **UserRegistrationData.swift**: Data structure to store and manage user registration information for users looking for roommates.
- **HaveRoomRegistrationData.swift**: Data structure for users who have a room to offer in the roommate-finding process.
- **ApartmentListingRegistrationData.swift**: Data structure for apartment listing registration, used by property managers.
- **Info.plist**: Configuration settings for the iOS app.
- **GoogleService-Info.plist**: Firebase configuration for backend integration.
- **RoommateSwipe.entitlements**: App entitlements for services like iCloud, push notifications, etc.
- **AllCode.txt**: A compilation of code references and snippets for development purposes.

### Models

- **Profile.swift**: Core data model representing a user's profile in the roommate matching system, including personal details and preferences.
- **UserProfile.swift**: Extended user profile information for authentication and registration.
- **ApartmentModels.swift**: Comprehensive data models for the apartment listing system, including apartment listings, interested users, messages, and status tracking.

### ViewModels

- **RoommateViewModel.swift**: The central view model that manages app state, data flow, and business logic for both roommate matching and apartment listings.

### Services

- **FirebaseService.swift**: Service layer that handles all Firebase interactions, including authentication, data storage, and retrieval for both roommate and apartment listing features.

### Views

#### Core Navigation

- **LoginView.swift**: The login screen with options for regular login, account creation, and the new apartment listing access.
- **RootTabView.swift**: Tab navigation for roommate-seeking users, showing browse, map, liked, matches, and profile tabs.
- **ApartmentRootTabView.swift**: Custom tab navigation for apartment listers with listings, interested users, analytics, and settings.
- **LoadingView.swift**: Loading screen displayed during app initialization.

#### Roommate Search Views

- **SwipeView.swift**: Main interface for browsing potential roommates with a card-swiping interface.
- **SwipeCardView.swift**: Individual card component for the swipe view, showing profile highlights.
- **MapView.swift**: Map-based view of available roommates or properties.
- **LikedView.swift**: Grid view of profiles the user has liked.
- **MatchesView.swift**: List of successful matches with messaging capabilities.
- **MatchedProfileView.swift**: Detailed view of a matched profile with chat functionality.

#### Profile Management

- **ProfileView.swift**: Basic profile overview screen.
- **ProfileDisplayView.swift**: Comprehensive profile display for viewing user information.
- **ProfileEditView.swift**: Interface for editing profile details.
- **DetailedProfileView.swift**: In-depth view of user profiles with additional information.
- **PhotoUploadView.swift**: Interface for uploading and managing profile and property photos.

#### Apartment Listing Views

- **ListingsView.swift**: Dashboard for property managers to view and manage their apartment listings.
- **InterestedUsersView.swift**: Interface showing users interested in listed properties, with filtering by status.
- **AnalyticsView.swift**: Data visualization and insights for property performance.
- **ApartmentSettingsView.swift**: Settings and account management for property owners.

#### Listing Detail Views

- **ListingDetailView.swift**: Detailed view of an apartment listing with property information and management options.
- **ListingEditView.swift**: Interface for editing apartment listing details.
- **ListingMapView.swift**: Map component for displaying property locations.
- **DetailedPropertyView.swift**: Comprehensive property details for roommate listings.
- **MapPropertyView.swift**: Map interface focused on property locations.

#### Registration & Onboarding

- **SignUpView.swift**: Initial registration view for new users.
- **ProfileTypeSelectionView.swift**: Selection screen for account type (looking for room, have room, or apartment lister).
- **BasicInfoView.swift**: Form for collecting basic user information during signup.
- **LifestyleQuestionsView.swift**: Interface for gathering lifestyle preferences for better matching.
- **PreferencesView.swift**: Form for setting general preferences.
- **PropertyPreferencesView.swift**: Form for setting property-specific preferences.
- **HaveRoomSignUpView.swift**: Registration flow for users offering a room.
- **ApartmentListingSignUpView.swift**: Registration flow for property managers.
- **ScheduleView.swift**: Interface for setting availability and scheduling.

#### UI Components

- **FormComponents.swift**: Reusable form elements and styles.
- **ButtonStyles.swift**: Custom button styles for consistent UI.
- **ChatComponents.swift**: Components for the messaging system.

## Technology Stack

- **Frontend**: SwiftUI for iOS app development
- **Backend**: Firebase for authentication, data storage, and real-time features
- **Data Model**: Swift structs with Codable for Firebase integration
- **Authentication**: Firebase Authentication
- **Storage**: Firebase Firestore for structured data, Firebase Storage for images
- **Analytics**: Firebase Analytics (integrated with custom analytics views)
- **Deployment**: iOS App Store (future)

## Future Enhancements

- Chat improvements with real-time messaging
- Advanced filtering options for apartment searches
- In-app scheduling for property viewings
- Payment integration for security deposits or application fees
- Push notifications for matches and messages