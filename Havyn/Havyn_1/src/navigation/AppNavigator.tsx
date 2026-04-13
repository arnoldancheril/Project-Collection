import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { useAuth } from '../context/AuthContext';

// Screens
import LoginScreen from '../screens/login/LoginScreen';
import RegisterScreen from '../screens/login/RegisterScreen';
import ProfileSetupScreen from '../screens/login/ProfileSetupScreen';
import MainTabNavigator from './MainTabNavigator';
import LoadingScreen from '../screens/LoadingScreen';

// Define stack navigator types
type RootStackParamList = {
  Login: undefined;
  Register: undefined;
  ProfileSetup: undefined;
  Main: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();

const AppNavigator = () => {
  const { user, userProfile, loading } = useAuth();

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!user ? (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Register" component={RegisterScreen} />
          </>
        ) : !userProfile ? (
          <Stack.Screen name="ProfileSetup" component={ProfileSetupScreen} />
        ) : (
          <Stack.Screen name="Main" component={MainTabNavigator} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator; 