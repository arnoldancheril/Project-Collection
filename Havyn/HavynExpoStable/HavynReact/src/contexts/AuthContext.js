import React, { createContext, useState, useEffect, useContext } from 'react';
import { onAuthStateChanged } from 'firebase/auth';
import { auth } from '../config/firebaseConfig';
import authService from '../services/AuthService';
import UserProfile from '../models/UserProfile';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [initialized, setInitialized] = useState(false);
  
  useEffect(() => {
    // For development: Skip Firebase auth state and initialize immediately
    setLoading(false);
    setInitialized(true);
    
    // Return an empty unsubscribe function
    return () => {};
  }, []);
  
  const register = async (email, password, userProfile) => {
    try {
      const result = await authService.register(email, password, userProfile);
      // No need to update state, onAuthStateChanged will handle it
      return result;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  };
  
  const login = async (email, password) => {
    try {
      const result = await authService.login(email, password);
      
      // Create a mock user profile for development
      const mockProfile = new UserProfile({
        id: result.user.uid,
        email: result.user.email,
        fullName: result.user.displayName,
        profileImageUrl: null,
        userType: email === 'admin' ? 'propertyManager' : 'roommate',
        // Add other profile fields as needed
        bio: 'This is a development test account',
        dateOfBirth: new Date('1990-01-01'),
        gender: 'Not specified',
        occupation: 'Developer',
        contactPhone: '555-123-4567',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastActive: new Date()
      });
      
      // Set auth state manually for development
      setCurrentUser(result.user);
      setUserProfile(mockProfile);
      
      return result;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };
  
  const logout = async () => {
    try {
      await authService.logout();
      // No need to update state, onAuthStateChanged will handle it
    } catch (error) {
      console.error('Logout error:', error);
      throw error;
    }
  };
  
  const updateProfile = async (profileData) => {
    try {
      await authService.updateUserProfile(profileData);
      // Refresh the profile
      const updatedProfile = await authService.getCurrentUserProfile();
      setUserProfile(updatedProfile);
      return updatedProfile;
    } catch (error) {
      console.error('Profile update error:', error);
      throw error;
    }
  };
  
  const resetPassword = async (email) => {
    try {
      return await authService.resetPassword(email);
    } catch (error) {
      console.error('Password reset error:', error);
      throw error;
    }
  };
  
  const changePassword = async (currentPassword, newPassword) => {
    try {
      return await authService.changePassword(currentPassword, newPassword);
    } catch (error) {
      console.error('Password change error:', error);
      throw error;
    }
  };
  
  const deleteAccount = async (password) => {
    try {
      return await authService.deleteAccount(password);
      // No need to update state, onAuthStateChanged will handle it
    } catch (error) {
      console.error('Account deletion error:', error);
      throw error;
    }
  };
  
  const getCurrentUserProfile = async () => {
    // Use the mock profile for development
    if (currentUser) {
      try {
        return await authService.getCurrentUserProfile();
      } catch (error) {
        console.error('Error getting user profile:', error);
        // Return a mock profile if there's an error
        return new UserProfile({
          id: currentUser.uid,
          email: currentUser.email,
          fullName: currentUser.displayName,
          userType: currentUser.email === 'admin' ? 'propertyManager' : 'roommate',
          // Add other necessary fields
          bio: 'Development account'
        });
      }
    }
    return null;
  };
  
  const value = {
    currentUser,
    userProfile,
    loading,
    initialized,
    register,
    login,
    logout,
    updateProfile,
    resetPassword,
    changePassword,
    deleteAccount,
    getCurrentUserProfile,
  };
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 