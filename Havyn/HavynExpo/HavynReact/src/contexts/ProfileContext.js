import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import SAMPLE_PROFILES from '../utils/sampleProfiles';

const ProfileContext = createContext(null);

export const ProfileProvider = ({ children }) => {
  const [profile, setProfile] = useState({
    name: 'Your Name',
    email: 'email@example.com',
    avatar: require('../../assets/person-placeholder.jpg'),
    preferences: {
      budget: {
        min: 800,
        max: 1500
      },
      moveInDate: new Date(),
      locations: ['River North', 'Lincoln Park', 'Wicker Park'],
      propertyType: ['Apartment', 'Condo'],
      roommates: {
        gender: 'Any',
        ageRange: [20, 35],
        cleanlinessLevel: 'Average',
        smokingAllowed: false,
        petsAllowed: true
      }
    }
  });
  
  // State for liked and matched profiles
  const [likedProfiles, setLikedProfiles] = useState([]);
  const [matchedProfiles, setMatchedProfiles] = useState([]);

  // Load saved profiles from AsyncStorage on component mount
  useEffect(() => {
    const loadStoredProfiles = async () => {
      try {
        const storedLikedProfiles = await AsyncStorage.getItem('likedProfiles');
        const storedMatchedProfiles = await AsyncStorage.getItem('matchedProfiles');
        
        if (storedLikedProfiles) {
          setLikedProfiles(JSON.parse(storedLikedProfiles));
        }
        
        if (storedMatchedProfiles) {
          setMatchedProfiles(JSON.parse(storedMatchedProfiles));
        }
      } catch (error) {
        console.error('Error loading stored profiles:', error);
      }
    };
    
    loadStoredProfiles();
  }, []);

  // Save profiles to AsyncStorage whenever they change
  useEffect(() => {
    const saveProfiles = async () => {
      try {
        await AsyncStorage.setItem('likedProfiles', JSON.stringify(likedProfiles));
        await AsyncStorage.setItem('matchedProfiles', JSON.stringify(matchedProfiles));
      } catch (error) {
        console.error('Error saving profiles:', error);
      }
    };
    
    saveProfiles();
  }, [likedProfiles, matchedProfiles]);

  const updateProfile = (newData) => {
    setProfile(prev => ({
      ...prev,
      ...newData
    }));
  };

  const updatePreferences = (newPreferences) => {
    setProfile(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        ...newPreferences
      }
    }));
  };

  // Function to like a profile
  const likeProfile = (profileToLike) => {
    // Check if profile is already liked to avoid duplicates
    if (!likedProfiles.some(p => p.id === profileToLike.id)) {
      const updatedLikedProfiles = [...likedProfiles, profileToLike];
      setLikedProfiles(updatedLikedProfiles);
      
      // Simulate a match with 30% probability
      if (Math.random() < 0.3 && !matchedProfiles.some(p => p.id === profileToLike.id)) {
        const updatedMatchedProfiles = [...matchedProfiles, profileToLike];
        setMatchedProfiles(updatedMatchedProfiles);
        return { isMatch: true };
      }
    }
    return { isMatch: false };
  };

  // Function to dislike a profile
  const dislikeProfile = (profileToDislike) => {
    setLikedProfiles(likedProfiles.filter(p => p.id !== profileToDislike.id));
    setMatchedProfiles(matchedProfiles.filter(p => p.id !== profileToDislike.id));
  };

  // Function to remove a profile from liked profiles
  const removeLiked = (profileId) => {
    setLikedProfiles(likedProfiles.filter(p => p.id !== profileId));
  };

  // Function to remove a profile from matched profiles
  const removeMatched = (profileId) => {
    setMatchedProfiles(matchedProfiles.filter(p => p.id !== profileId));
  };

  const value = {
    profile,
    updateProfile,
    updatePreferences,
    likedProfiles,
    matchedProfiles,
    likeProfile,
    dislikeProfile,
    removeLiked,
    removeMatched
  };

  return (
    <ProfileContext.Provider value={value}>
      {children}
    </ProfileContext.Provider>
  );
};

export const useProfile = () => {
  const context = useContext(ProfileContext);
  if (context === null) {
    throw new Error('useProfile must be used within a ProfileProvider');
  }
  return context;
}; 