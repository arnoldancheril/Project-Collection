# Havyn - React Native Roommate Finder

Havyn is a mobile application that helps users find the perfect roommate match. This is the React Native implementation of the original Swift application.

## Features

- User authentication (login, register, forgot password)
- Onboarding flow for new users
- Profile type selection (looking for room, have a room, property manager)
- Swipe interface for finding roommates
- Detailed profile view for potential matches
- Message system for connected users
- Property listings management
- User profile management

## Tech Stack

- React Native
- Expo
- Firebase (Authentication, Firestore, Storage)
- React Navigation

## Getting Started

### Prerequisites

- Node.js (v14.0.0 or higher)
- npm (v7.0.0 or higher) or yarn (v1.22.0 or higher)
- Expo CLI (`npm install -g expo-cli`)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/havyn-react.git
cd havyn-react
```

2. Install dependencies
```bash
npm install
# or
yarn install
```

3. Set up Firebase credentials
- Create a Firebase project at [https://console.firebase.google.com/](https://console.firebase.google.com/)
- Enable Authentication (Email/Password), Firestore, and Storage services
- Copy your Firebase config from the Firebase console
- Paste the config in `src/config/firebaseConfig.js`

### Running the App

```bash
npm start
# or
yarn start
```

This will start the Expo development server. You can run the app on:
- iOS Simulator (Mac only)
- Android Emulator
- Physical device using the Expo Go app (scan the QR code)

#### Specific Platforms

For iOS:
```bash
npm run ios
# or 
yarn ios
```

For Android:
```bash
npm run android
# or
yarn android
```

## Testing

To test Expo features and API integrations, navigate to the Test Screen:
1. Open the app and login
2. Tap on "Test Expo Features" button on the login screen (in development mode)
3. Run various tests to verify functionality

## Project Structure

```
HavynReact/
├── assets/            # Static assets like images and icons
├── src/
│   ├── components/    # Reusable UI components
│   ├── config/        # Configuration files
│   ├── constants/     # Constants and app-wide variables
│   ├── contexts/      # React Context providers
│   ├── hooks/         # Custom React hooks
│   ├── models/        # Data models and types
│   ├── navigation/    # Navigation configurations
│   ├── screens/       # App screens
│   ├── services/      # API and service integrations
│   └── utils/         # Utility functions
├── App.js             # Main App component
└── index.js           # Entry point
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Swift application developers
- Expo and React Native communities
- Firebase platform

## Troubleshooting

### Reanimated Issues

If you see the error "Native part of Reanimated doesn't seem to be initialized", try:

1. Make sure your babel.config.js includes the plugin:
```js
module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      'react-native-reanimated/plugin',
    ],
  };
};
```

2. Ensure "newArchEnabled" is set to true in app.json:
```json
{
  "expo": {
    "newArchEnabled": true,
    ...
  }
}
```

3. Clear Metro cache and restart:
```bash
npm start -- --clear
```

### Firebase Authentication Issues

If you see Firebase Auth warnings about AsyncStorage:

1. Make sure @react-native-async-storage/async-storage is installed:
```bash
npm install @react-native-async-storage/async-storage
```

2. Configure Firebase Auth to use AsyncStorage:
```js
import { initializeAuth, getReactNativePersistence } from 'firebase/auth';
import AsyncStorage from '@react-native-async-storage/async-storage';

export const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage)
});
```

### Package Compatibility Issues

If you see warnings about package versions:

1. Check the Expo version compatibility table: https://docs.expo.dev/versions/latest/sdk/overview/
2. Update packages to versions compatible with your Expo SDK:
```bash
npx expo install react-native-reanimated@3.16.1
```

## Documentation

For more detailed information, refer to `PROJECT_DOCUMENTATION.md` in this repository.