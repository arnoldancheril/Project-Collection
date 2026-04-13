import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Screens
import LoginScreen from '../screens/auth/LoginScreen';
import SignupScreen from '../screens/auth/SignupScreen';
import ForgotPasswordScreen from '../screens/auth/ForgotPasswordScreen';
import ProfileTypeScreen from '../screens/auth/ProfileTypeScreen';
import BasicInfoScreen from '../screens/auth/BasicInfoScreen';

const Stack = createNativeStackNavigator();

const AuthStack = () => {
  return (
    <Stack.Navigator
      initialRouteName="Login"
      screenOptions={{
        headerShown: false,
      }}
    >
      <Stack.Screen
        name="Login"
        component={LoginScreen}
      />
      <Stack.Screen
        name="Signup"
        component={SignupScreen}
      />
      <Stack.Screen
        name="ForgotPassword"
        component={ForgotPasswordScreen}
      />
      <Stack.Screen
        name="ProfileType"
        component={ProfileTypeScreen}
        options={{ headerBackVisible: false }}
      />
      <Stack.Screen
        name="BasicInfo"
        component={BasicInfoScreen}
      />
    </Stack.Navigator>
  );
};

export default AuthStack; 