import { Timestamp } from 'firebase/firestore';

export type Gender = 'male' | 'female' | 'non-binary' | 'prefer_not_to_say';
export type ProfileType = 'looking_for_room' | 'have_room' | 'apartment_listing';
export type CleanlinessLevel = 1 | 2 | 3 | 4 | 5;
export type NoiseLevel = 1 | 2 | 3 | 4 | 5;
export type SocialLevel = 1 | 2 | 3 | 4 | 5;
export type SleepSchedule = 'early_bird' | 'night_owl' | 'regular';

export interface UserPreferences {
  cleanliness: CleanlinessLevel;
  noiseLevel: NoiseLevel;
  socialLevel: SocialLevel;
  sleepSchedule: SleepSchedule;
  preferredRoommateGender?: Gender | 'any';
  preferredAgeRange?: {
    min: number;
    max: number;
  };
  monthlyRentBudget?: number; // Only required if profileType is "looking_for_room"
}

export interface UserImages {
  profile: string[]; // Array of profile image URLs (2-5 images)
  property?: string[]; // Array of property image URLs (for users with rooms)
}

export interface User {
  id: string; // auth.currentUser.uid
  userId?: string; // Sequential user ID for easy identification (00001, 00002, etc.)
  email: string;
  name: string;
  birthday: Timestamp;
  age?: number; // Calculated field
  gender: Gender;
  profileType: ProfileType;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  profileImageUrl?: string; // Legacy: Main profile image URL (for backward compatibility)
  images?: UserImages; // New: Organized image structure
  preferences: UserPreferences;
  descriptions: string[]; // Array of 3 strings
  habitsSummary?: string;
  lookingForSummary?: string;
} 