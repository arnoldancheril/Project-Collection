import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';

// Import actual screen components
import LoadingScreen from '../screens/auth/LoadingScreen';
import LoginScreen from '../screens/auth/LoginScreen';
import SignUpScreen from '../screens/auth/SignUpScreen';
import HomeScreen from '../screens/main/HomeScreen';
import MapScreen from '../screens/main/MapScreen';
import MatchesScreen from '../screens/main/MatchesScreen';
import MessagesScreen from '../screens/main/MessagesScreen';
import ProfileScreen from '../screens/main/ProfileScreen';
import { colors, spacing } from '../styles/theme';

// Define the types for our navigation
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
};

export type AuthStackParamList = {
  Loading: undefined;
  Login: undefined;
  SignUp: undefined;
};

export type MainTabParamList = {
  Home: undefined;
  Map: undefined;
  Matches: undefined;
  Messages: undefined;
  Profile: undefined;
};

// Create the navigators
const RootStack = createStackNavigator<RootStackParamList>();
const AuthStack = createStackNavigator<AuthStackParamList>();
const MainTab = createBottomTabNavigator<MainTabParamList>();

// Loading screen wrapper that automatically navigates to Login after 3 seconds
const LoadingScreenWrapper = ({ navigation }: any) => (
  <LoadingScreen onFinish={() => navigation.replace('Login')} />
);

// Auth Navigator
const AuthNavigator = () => {
  return (
    <AuthStack.Navigator
      screenOptions={{
        headerShown: false,
      }}
      initialRouteName="Loading"
    >
      <AuthStack.Screen name="Loading" component={LoadingScreenWrapper} />
      <AuthStack.Screen name="Login" component={LoginScreen} />
      <AuthStack.Screen name="SignUp" component={SignUpScreen} />
    </AuthStack.Navigator>
  );
};

// Main Tab Navigator
const MainNavigator = () => {
  return (
    <MainTab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Map':
              iconName = focused ? 'map' : 'map-outline';
              break;
            case 'Matches':
              iconName = focused ? 'heart' : 'heart-outline';
              break;
            case 'Messages':
              iconName = focused ? 'chatbubbles' : 'chatbubbles-outline';
              break;
            case 'Profile':
              iconName = focused ? 'person' : 'person-outline';
              break;
            default:
              iconName = 'home-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: colors.primaryProfile,
        tabBarInactiveTintColor: colors.text.secondary,
        tabBarStyle: {
          height: 72,
          paddingBottom: spacing.md,
          paddingTop: spacing.sm,
          borderRadius: 32,
          position: 'absolute',
          bottom: spacing.md,
          left: spacing.md,
          right: spacing.md,
          backgroundColor: colors.white,
          borderTopWidth: 0,
          shadowColor: colors.primaryProfile,
          shadowOffset: { width: 0, height: 4 },
          shadowOpacity: 0.1,
          shadowRadius: 8,
          elevation: 8,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '400',
        },
        tabBarItemStyle: {
          paddingVertical: spacing.xs,
        },
      })}
    >
      <MainTab.Screen name="Home" component={HomeScreen} options={{ title: 'Home' }} />
      <MainTab.Screen name="Map" component={MapScreen} options={{ title: 'Map' }} />
      <MainTab.Screen name="Matches" component={MatchesScreen} options={{ title: 'Matches' }} />
      <MainTab.Screen name="Messages" component={MessagesScreen} options={{ title: 'Messages' }} />
      <MainTab.Screen name="Profile" component={ProfileScreen} options={{ title: 'Profile' }} />
    </MainTab.Navigator>
  );
};

// Root Navigator
export const RootNavigator = () => {
  return (
    <NavigationContainer>
      <RootStack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
        <RootStack.Screen name="Auth" component={AuthNavigator} />
        <RootStack.Screen name="Main" component={MainNavigator} />
      </RootStack.Navigator>
    </NavigationContainer>
  );
}; 