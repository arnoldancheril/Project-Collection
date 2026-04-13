import React from 'react';
import { View, Platform, Dimensions } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../utils/theme';

// Main Tab Screens
import SwipeScreen from '../screens/roommate/SwipeScreen';
import MapScreen from '../screens/roommate/MapScreen';
import DetailedProfileScreen from '../screens/roommate/DetailedProfileScreen';
import SettingsScreen from '../screens/roommate/SettingsScreen';
import PropertyDetailsScreen from '../screens/roommate/PropertyDetailsScreen';

// Placeholder screens - create minimal versions for navigation to work
import LikedScreen from '../screens/roommate/LikedScreen';
import MatchesScreen from '../screens/roommate/MatchesScreen';
import ProfileScreen from '../screens/roommate/ProfileScreen';

// Import the new screens
import GroupChatDetail from '../screens/messages/GroupChatDetail';
import CreateGroupChat from '../screens/messages/CreateGroupChat';
import ChatScreen from '../screens/messages/ChatScreen';

const Tab = createBottomTabNavigator();
const RoommateScreenStack = createNativeStackNavigator();
const SwipeStack = createNativeStackNavigator();
const MapStack = createNativeStackNavigator();
const LikedStack = createNativeStackNavigator();
const MatchesStack = createNativeStackNavigator();
const ProfileStack = createNativeStackNavigator();

// Stack navigators for each tab
const SwipeStackScreen = () => (
  <SwipeStack.Navigator>
    <SwipeStack.Screen 
      name="SwipeMain" 
      component={SwipeScreen} 
      options={{ headerShown: false }}
    />
    <SwipeStack.Screen 
      name="DetailedProfile" 
      component={DetailedProfileScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
        animationTypeForReplace: 'push',
      }}
    />
    <SwipeStack.Screen 
      name="Settings" 
      component={SettingsScreen} 
      options={{ 
        title: 'Settings',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
  </SwipeStack.Navigator>
);

const MapStackScreen = () => (
  <MapStack.Navigator>
    <MapStack.Screen 
      name="MapMain" 
      component={MapScreen} 
      options={{ 
        title: 'Property Map',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      }}
    />
    <MapStack.Screen 
      name="PropertyDetails" 
      component={PropertyDetailsScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
      }}
    />
  </MapStack.Navigator>
);

const LikedStackScreen = () => (
  <LikedStack.Navigator>
    <LikedStack.Screen 
      name="LikedMain" 
      component={LikedScreen} 
      options={{ 
        title: 'Liked Profiles',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
    <LikedStack.Screen 
      name="DetailedProfile" 
      component={DetailedProfileScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
      }}
    />
  </LikedStack.Navigator>
);

const MatchesStackScreen = () => (
  <MatchesStack.Navigator>
    <MatchesStack.Screen 
      name="MatchesMain" 
      component={MatchesScreen} 
      options={{ 
        title: 'Your Matches',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
    <MatchesStack.Screen 
      name="DetailedProfile" 
      component={DetailedProfileScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
      }}
    />
    <MatchesStack.Screen
      name="ChatScreen"
      component={ChatScreen}
      options={{
        headerShown: false,
      }}
    />
    <MatchesStack.Screen
      name="GroupChatDetail"
      component={GroupChatDetail}
      options={{
        headerShown: false,
      }}
    />
    <MatchesStack.Screen
      name="CreateGroupChat"
      component={CreateGroupChat}
      options={{
        headerShown: false,
        presentation: 'modal',
      }}
    />
  </MatchesStack.Navigator>
);

const ProfileStackScreen = () => (
  <ProfileStack.Navigator>
    <ProfileStack.Screen 
      name="ProfileMain" 
      component={ProfileScreen} 
      options={{ 
        title: 'Your Profile',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
    <ProfileStack.Screen 
      name="Settings" 
      component={SettingsScreen} 
      options={{ 
        title: 'Settings',
        headerStyle: {
          backgroundColor: COLORS.primary,
        },
        headerTintColor: '#fff',
      }}
    />
    <ProfileStack.Screen 
      name="PropertyDetails" 
      component={PropertyDetailsScreen} 
      options={{ 
        headerShown: false,
        presentation: 'modal',
      }}
    />
  </ProfileStack.Navigator>
);

// Custom tab bar icon with indicator
const TabBarIcon = ({ focused, name, color }) => {
  return (
    <View style={{ 
      alignItems: 'center',
      justifyContent: 'center',
      width: 60,
      height: 30
    }}>
      <Ionicons 
        name={name} 
        size={24} 
        color={color}
        style={{
          opacity: focused ? 1 : 0.8
        }}
      />
      {focused && (
        <View 
          style={{
            position: 'absolute',
            bottom: -5,
            width: 5,
            height: 5,
            borderRadius: 3,
            backgroundColor: COLORS.primary
          }}
        />
      )}
    </View>
  );
};

// Main Tab Navigator
const TabNavigator = () => {
  const { height } = Dimensions.get('window');
  const isIphoneX = Platform.OS === 'ios' && (height >= 812);

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Browse') {
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

          return <TabBarIcon focused={focused} name={iconName} color={color} />;
        },
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: '#9E9E9E',
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#fff',
          borderTopWidth: 1,
          borderTopColor: '#f0f0f0',
          height: isIphoneX ? 85 : 60,
          paddingTop: 5,
          paddingBottom: isIphoneX ? 30 : 10,
          shadowColor: '#000',
          shadowOffset: { width: 0, height: -2 },
          shadowOpacity: 0.05,
          shadowRadius: 3.84,
          elevation: 5,
          position: 'absolute',
          bottom: 0,
          left: 0, 
          right: 0,
          // Ensure safe area for notched devices
          paddingBottom: isIphoneX ? 34 : 10,
          height: isIphoneX ? 90 : 60,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
          marginTop: 0,
        },
        tabBarShowLabel: true,
        tabBarHideOnKeyboard: true,
      })}
    >
      <Tab.Screen 
        name="Browse" 
        component={SwipeStackScreen} 
        options={{
          tabBarLabel: 'Discover'
        }}
      />
      <Tab.Screen 
        name="Map" 
        component={MapStackScreen} 
        options={{
          tabBarLabel: 'Map'
        }}
      />
      <Tab.Screen 
        name="Liked" 
        component={LikedStackScreen} 
        options={{
          tabBarLabel: 'Liked'
        }}
      />
      <Tab.Screen 
        name="Matches" 
        component={MatchesStackScreen} 
        options={{
          tabBarLabel: 'Matches'
        }}
      />
      <Tab.Screen 
        name="Profile" 
        component={ProfileStackScreen} 
        options={{
          tabBarLabel: 'Profile'
        }}
      />
    </Tab.Navigator>
  );
};

// Root Stack for the Roommate flow
const RoommateStack = () => {
  return (
    <RoommateScreenStack.Navigator>
      <RoommateScreenStack.Screen
        name="Main"
        component={TabNavigator}
        options={{ headerShown: false }}
      />
    </RoommateScreenStack.Navigator>
  );
};

export default RoommateStack; 