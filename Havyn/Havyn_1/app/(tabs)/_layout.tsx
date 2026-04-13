import React from 'react';
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

export default function TabLayout() {
  return (
    <Tabs screenOptions={{ tabBarActiveTintColor: '#3498db' }}>
      <Tabs.Screen 
        name="index" 
        options={{
          title: "Home",
          tabBarIcon: ({ color, size }) => <Ionicons name="home" size={size} color={color} />
        }}
      />
      <Tabs.Screen 
        name="map" 
        options={{
          title: "Map",
          tabBarIcon: ({ color, size }) => <Ionicons name="map" size={size} color={color} />
        }}
      />
      <Tabs.Screen 
        name="matches" 
        options={{
          title: "Matches",
          tabBarIcon: ({ color, size }) => <Ionicons name="heart" size={size} color={color} />
        }}
      />
      <Tabs.Screen 
        name="groups" 
        options={{
          title: "Groups",
          tabBarIcon: ({ color, size }) => <Ionicons name="people" size={size} color={color} />
        }}
      />
      <Tabs.Screen 
        name="profile" 
        options={{
          title: "My Profile",
          tabBarIcon: ({ color, size }) => <Ionicons name="person" size={size} color={color} />
        }}
      />
    </Tabs>
  );
} 