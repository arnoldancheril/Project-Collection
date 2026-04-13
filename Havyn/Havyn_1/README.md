# Havyn - Chicago Roommate Finder

A mobile application built with React Native and Expo that connects people looking for roommates in Chicago.

## Setup Instructions

1. Clone the repository
2. Install dependencies: `npm install --legacy-peer-deps`
3. Start the development server: `npx expo start`

## Known Issues and Fixes

### Firebase Authentication Issue

The application had an issue with "Component auth has not been registered yet" error when using Firebase authentication. 

**Fix:**
- Renamed Firebase authentication imports to avoid conflicts with reserved keywords:
  - Changed `auth` to `firebaseAuthentication` in firebaseConfig.js
  - Updated all imports and references in AuthContext.tsx and firebaseService.ts

**Current Status:**
- Basic UI is working
- Authentication is disabled until the import conflicts are fully resolved

## Development Guide

To add back authentication functionality:
1. Ensure all Firebase authentication references use `firebaseAuthentication` consistently
2. Add back the AuthProvider and AppNavigator components one by one
3. Test each step to identify any remaining naming conflicts

## Technology Stack
- **Frontend**: React Native, Expo
- **Backend**: Firebase (Authentication, Firestore, Storage)
- **State Management**: React Context API
- **Navigation**: React Navigation
