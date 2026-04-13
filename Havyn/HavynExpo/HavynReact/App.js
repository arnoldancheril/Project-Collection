import React, { useState, useEffect } from 'react';
import { 
  StatusBar, 
  SafeAreaView,
  StyleSheet, 
  View, 
  Text
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { ProfileProvider } from './src/contexts/ProfileContext';
import SwipeScreen from './src/screens/roommate/SwipeScreen';
import MapScreen from './src/screens/roommate/MapScreen';
import DetailedProfileScreen from './src/screens/roommate/DetailedProfileScreen';
import PropertyDetailsScreen from './src/screens/roommate/PropertyDetailsScreen';
import MessagesScreen from './src/screens/messages/MessagesScreen';
import ChatScreen from './src/screens/messages/ChatScreen';
import GroupChatDetail from './src/screens/messages/GroupChatDetail';
import CreateGroupChat from './src/screens/messages/CreateGroupChat';
import ProfileScreen from './src/screens/profile/ProfileScreen';
import SplashScreen from './src/screens/auth/SplashScreen';
import LikedScreen from './src/screens/roommate/LikedScreen';
import MatchesScreen from './src/screens/roommate/MatchesScreen';
import SettingsScreen from './src/screens/roommate/SettingsScreen';
import { COLORS } from './src/utils/theme';

// Create navigators
const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

// Bottom tab navigation
const TabNavigator = () => (
  <Tab.Navigator
    screenOptions={({ route }) => ({
      tabBarIcon: ({ focused, color, size }) => {
        let iconName;

        if (route.name === 'Discover') {
          iconName = focused ? 'grid' : 'grid-outline';
        } else if (route.name === 'Map') {
          iconName = focused ? 'map' : 'map-outline';
        } else if (route.name === 'Liked') {
          iconName = focused ? 'heart' : 'heart-outline';
        } else if (route.name === 'Matches') {
          iconName = focused ? 'people' : 'people-outline';
        } else if (route.name === 'Profile') {
          iconName = focused ? 'person-circle' : 'person-circle-outline';
        }

        return <Ionicons name={iconName} size={size} color={color} />;
      },
      tabBarActiveTintColor: COLORS.primary,
      tabBarInactiveTintColor: 'gray',
      tabBarStyle: {
        borderTopWidth: 1,
        borderTopColor: '#f0f0f0',
        height: 60,
        paddingBottom: 8,
        paddingTop: 5,
      },
      tabBarLabelStyle: {
        fontSize: 12,
        fontWeight: '500',
      },
      headerShown: false,
    })}
  >
    <Tab.Screen name="Discover" component={DiscoverStack} />
    <Tab.Screen name="Map" component={MapStack} />
    <Tab.Screen name="Liked" component={LikedStack} />
    <Tab.Screen name="Matches" component={MatchesStack} />
    <Tab.Screen name="Profile" component={ProfileStack} />
  </Tab.Navigator>
);

// Stack for Discover tab
const DiscoverStack = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: false,
    }}
  >
    <Stack.Screen name="SwipeScreen" component={SwipeScreen} />
    <Stack.Screen 
      name="DetailedProfile" 
      component={DetailedProfileScreen} 
      options={{ 
        headerShown: false, 
        presentation: 'modal' 
      }} 
    />
    <Stack.Screen 
      name="Settings" 
      component={SettingsScreen} 
      options={{ 
        headerShown: true,
        title: 'Settings',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
  </Stack.Navigator>
);

// Stack for Map tab
const MapStack = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: false,
    }}
  >
    <Stack.Screen name="MapScreen" component={MapScreen} />
    <Stack.Screen 
      name="PropertyDetails" 
      component={PropertyDetailsScreen} 
      options={{
        headerShown: false,
        presentation: 'modal'
      }}
    />
  </Stack.Navigator>
);

// Stack for Liked tab
const LikedStack = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: true,
    }}
  >
    <Stack.Screen 
      name="LikedScreen" 
      component={LikedScreen} 
      options={{ 
        title: 'Liked Profiles',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
    <Stack.Screen 
      name="DetailedProfile" 
      component={DetailedProfileScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
      }}
    />
  </Stack.Navigator>
);

// Stack for Matches tab
const MatchesStack = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: true,
    }}
  >
    <Stack.Screen 
      name="MatchesScreen" 
      component={MatchesScreen} 
      options={{ 
        title: 'Your Matches',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
    <Stack.Screen 
      name="DetailedProfile" 
      component={DetailedProfileScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
      }}
    />
    <Stack.Screen 
      name="ChatScreen" 
      component={ChatScreen}
      options={{
        headerShown: false
      }}
    />
    <Stack.Screen 
      name="GroupChatDetail" 
      component={GroupChatDetail}
      options={{
        headerShown: false
      }}
    />
    <Stack.Screen 
      name="CreateGroupChat" 
      component={CreateGroupChat}
      options={{
        headerShown: false,
        presentation: 'modal'
      }}
    />
  </Stack.Navigator>
);

// Stack for Profile tab
const ProfileStack = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: false,
    }}
  >
    <Stack.Screen name="ProfileScreen" component={ProfileScreen} />
    <Stack.Screen 
      name="Settings" 
      component={SettingsScreen} 
      options={{ 
        headerShown: true,
        title: 'Settings',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
  </Stack.Navigator>
);

const App = () => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <SplashScreen />;
  }

  return (
    <ProfileProvider>
      <NavigationContainer>
        <StatusBar barStyle="dark-content" backgroundColor="#fff" />
        <TabNavigator />
      </NavigationContainer>
    </ProfileProvider>
  );
};

export default App;
