import { Redirect } from 'expo-router';

export default function Index() {
  // Redirect to the auth login page
  return <Redirect href="/(auth)/login" />;
} 