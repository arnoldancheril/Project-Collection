import React, { createContext, useState, useEffect, useContext } from 'react';
import { User as FirebaseUser } from 'firebase/auth';
// import { firebaseAuthentication } from '../../firebaseConfig'; // Temporarily disabled
import { getUserProfile } from '../services/firebaseService';
import { User } from '../models';

interface AuthContextType {
  user: FirebaseUser | null;
  userProfile: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [userProfile, setUserProfile] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // AUTH TEMPORARILY DISABLED - Firebase authentication is commented out
    // When re-enabling, uncomment the firebaseAuthentication import and this code:
    
    /*
    const unsubscribe = firebaseAuthentication.onAuthStateChanged(async (authUser) => {
      setUser(authUser);
      
      if (authUser) {
        // Load user profile from Firestore
        const profile = await getUserProfile(authUser.uid);
        setUserProfile(profile);
      } else {
        setUserProfile(null);
      }
      
      setLoading(false);
    });

    return () => unsubscribe();
    */
    
    // Temporary: Just set loading to false since auth is disabled
    setLoading(false);
  }, []);

  const signIn = async (email: string, password: string) => {
    // Placeholder - auth disabled
    throw new Error('Authentication temporarily disabled');
  };

  const signUp = async (email: string, password: string) => {
    // Placeholder - auth disabled
    throw new Error('Authentication temporarily disabled');
  };

  const signOut = async () => {
    // Placeholder - auth disabled
    throw new Error('Authentication temporarily disabled');
  };

  const value = {
    user,
    userProfile,
    loading,
    signIn,
    signUp,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}; 