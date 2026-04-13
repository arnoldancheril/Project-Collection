// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// import { initializeAuth, getReactNativePersistence } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
// import ReactNativeAsyncStorage from '@react-native-async-storage/async-storage';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: (
    process.env.EXPO_PUBLIC_FIREBASE_API_KEY ||
    process.env.REACT_APP_FIREBASE_API_KEY ||
    process.env.NEXT_PUBLIC_FIREBASE_API_KEY ||
    process.env.FIREBASE_API_KEY ||
    ""
  ),
  authDomain: "havynbackend.firebaseapp.com",
  projectId: "havynbackend",
  storageBucket: "havynbackend.firebasestorage.app",
  messagingSenderId: "346096844265",
  appId: "1:346096844265:web:1ebcbbed0e79b315345f78",
  measurementId: "G-431S56WVJN"
};

if (!firebaseConfig.apiKey) {
  console.warn(
    "Firebase apiKey missing. Set EXPO_PUBLIC_FIREBASE_API_KEY / REACT_APP_FIREBASE_API_KEY / NEXT_PUBLIC_FIREBASE_API_KEY / FIREBASE_API_KEY in your environment."
  );
}

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Auth with AsyncStorage persistence (TEMPORARILY DISABLED)
// const firebaseAuthentication = initializeAuth(app, {
//   persistence: getReactNativePersistence(ReactNativeAsyncStorage)
// });

// Initialize other Firebase services
const db = getFirestore(app);
const storage = getStorage(app);

// Export Firebase services with different names to avoid conflicts
export { 
  // firebaseAuthentication, // Temporarily disabled to avoid auth registration error
  db, 
  storage, 
  app as firebase // Export the app instance
};
