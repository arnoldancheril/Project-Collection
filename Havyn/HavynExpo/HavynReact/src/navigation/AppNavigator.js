import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuth } from '../contexts/AuthContext';

// Stacks
import AuthStack from './AuthStack';
import RoommateStack from './RoommateStack';
import PropertyManagerStack from './PropertyManagerStack';
import LoadingScreen from '../screens/LoadingScreen';
import TestScreen from '../screens/testing/TestScreen';

const RootStack = createNativeStackNavigator();

const AppNavigator = () => {
  const { currentUser, userProfile, loading, initialized } = useAuth();
  
  if (!initialized) {
    return <LoadingScreen message="Initializing application..." />;
  }
  
  if (loading) {
    return <LoadingScreen message="Loading profile..." />;
  }
  
  return (
    <NavigationContainer>
      <RootStack.Navigator screenOptions={{ headerShown: false }}>
        {!currentUser ? (
          <RootStack.Screen name="Auth" component={AuthStack} />
        ) : !userProfile ? (
          <RootStack.Screen name="Loading" component={LoadingScreen} initialParams={{ message: "Setting up your profile..." }} />
        ) : userProfile.profileType === 'PROPERTY_MANAGER' ? (
          <RootStack.Screen name="PropertyManager" component={PropertyManagerStack} />
        ) : (
          <RootStack.Screen name="Roommate" component={RoommateStack} />
        )}
        
        {/* Test screen accessible from anywhere in the app via direct navigation */}
        <RootStack.Screen 
          name="Test" 
          component={TestScreen} 
          options={{ presentation: 'modal' }}
        />
      </RootStack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator; 