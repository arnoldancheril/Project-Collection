import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../utils/theme';

// Import only the screens that exist
import SettingsScreen from '../screens/propertyManager/SettingsScreen';

// Commented out imports for screens that don't exist yet
/*
import ListingsScreen from '../screens/propertyManager/ListingsScreen';
import InterestedUsersScreen from '../screens/propertyManager/InterestedUsersScreen';
import AnalyticsScreen from '../screens/propertyManager/AnalyticsScreen';
import ListingDetailsScreen from '../screens/propertyManager/ListingDetailsScreen';
import EditListingScreen from '../screens/propertyManager/EditListingScreen';
import AddListingScreen from '../screens/propertyManager/AddListingScreen';
import UserProfileScreen from '../screens/propertyManager/UserProfileScreen';
import EditProfileScreen from '../screens/propertyManager/EditProfileScreen';
import PropertyMapScreen from '../screens/propertyManager/PropertyMapScreen';
*/

const Tab = createBottomTabNavigator();
const PropertyManagerScreenStack = createNativeStackNavigator();
const SettingsStack = createNativeStackNavigator();

// Only keeping the stack that has screens that exist
const SettingsStackScreen = () => (
  <SettingsStack.Navigator>
    <SettingsStack.Screen 
      name="SettingsMain" 
      component={SettingsScreen} 
      options={{ title: 'Settings' }}
    />
    {/* Commented out screens that don't exist yet
    <SettingsStack.Screen 
      name="EditProfile" 
      component={EditProfileScreen} 
      options={{ title: 'Edit Profile' }}
    />
    */}
  </SettingsStack.Navigator>
);

// Simplified Tab Navigator with only existing screens
const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          if (route.name === 'Settings') {
            return <Ionicons name={focused ? 'settings' : 'settings-outline'} size={size} color={color} />;
          }
          return null;
        },
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: COLORS.textSecondary,
      })}
    >
      <Tab.Screen name="Settings" component={SettingsStackScreen} options={{ headerShown: false }} />
      
      {/* Commented out tabs that don't have screens yet 
      <Tab.Screen name="Listings" component={ListingsStackScreen} options={{ headerShown: false }} />
      <Tab.Screen name="Interested" component={InterestedStackScreen} options={{ headerShown: false }} />
      <Tab.Screen name="Analytics" component={AnalyticsStackScreen} options={{ headerShown: false }} />
      */}
    </Tab.Navigator>
  );
};

// Root Stack for the Property Manager flow
const PropertyManagerStack = () => {
  return (
    <PropertyManagerScreenStack.Navigator>
      <PropertyManagerScreenStack.Screen
        name="Main"
        component={TabNavigator}
        options={{ headerShown: false }}
      />
    </PropertyManagerScreenStack.Navigator>
  );
};

export default PropertyManagerStack; 