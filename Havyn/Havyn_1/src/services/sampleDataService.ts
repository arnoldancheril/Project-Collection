import { Timestamp } from 'firebase/firestore';
import { collection, addDoc, getDocs, doc, setDoc, writeBatch, query, orderBy, deleteDoc } from 'firebase/firestore';
import { db } from '../../firebaseConfig';
import { User, Gender, ProfileType } from '../models/User';
import { uploadSampleImagesToStorage, getStorageImageUrls, uploadSampleUsersImagesWithStructure, clearAllSampleUserImages, scanAllStorageUsers } from './imageService';

// Sample data arrays for generating realistic profiles
const names = [
  'Alex Johnson', 'Sarah Chen', 'Michael Rodriguez', 'Emily Davis', 'David Kim',
  'Jessica Wilson', 'Ryan Martinez', 'Ashley Thompson', 'Brandon Lee', 'Amanda Garcia',
  'Jordan Brown', 'Taylor Anderson', 'Casey Miller', 'Morgan Jones', 'Jamie Lewis',
  'Samantha Park', 'Chris Nakamura', 'Priya Patel', 'Marcus Williams', 'Emma Johnson',
  'Kai Zhang', 'Isabella Rodriguez', 'Noah Thompson', 'Zoe Williams', 'Ethan Davis',
  'Maya Singh', 'Lucas Brown', 'Olivia Martinez', 'Gabriel Kim', 'Sophia Chang',
  'Hunter Smith', 'Ava Wilson', 'Diego Morales', 'Chloe Anderson', 'Cameron Foster'
];

const descriptions = [
  [
    "I'm a grad student at UChicago studying economics.",
    "Love cooking and trying new restaurants around the city.",
    "Looking for someone chill who enjoys good conversations."
  ],
  [
    "Working professional in tech, remote most days.",
    "Big fan of hiking and outdoor activities on weekends.",
    "Prefer a clean, organized living space."
  ],
  [
    "Medical resident at Northwestern Memorial Hospital.",
    "Quiet person who values a peaceful home environment.",
    "Love reading and watching documentaries in my free time."
  ],
  [
    "Marketing coordinator for a local startup.",
    "Social butterfly who loves hosting dinner parties.",
    "Looking for someone who's open to hanging out together."
  ],
  [
    "Artist and freelance graphic designer.",
    "Night owl who works best in the evening hours.",
    "Need a creative, inspiring living space."
  ],
  [
    "Law student at Northwestern with a busy schedule.",
    "Early riser who enjoys morning workouts.",
    "Looking for a study-friendly, quiet environment."
  ],
  [
    "Software engineer at a Chicago tech company.",
    "Love gaming and building side projects in my spare time.",
    "Prefer someone who respects personal space and downtime."
  ],
  [
    "Nurse at Rush University Medical Center.",
    "Work rotating shifts, so flexible living situation needed.",
    "Enjoy yoga, cooking, and weekend farmers markets."
  ],
  [
    "PhD student in neuroscience at University of Chicago.",
    "Passionate about research and science communication.",
    "Love board games and craft beer on weekends."
  ],
  [
    "Financial analyst working in the Loop.",
    "Enjoy running along the lakefront and cycling.",
    "Looking for someone who appreciates a good work-life balance."
  ],
  [
    "Elementary school teacher in Lincoln Park.",
    "Enjoy arts and crafts, reading, and volunteering.",
    "Want a roommate who's patient and understanding."
  ],
  [
    "Bartender at a popular River North restaurant.",
    "Work late nights but love exploring the city during the day.",
    "Looking for someone flexible with different schedules."
  ],
  [
    "Personal trainer and fitness enthusiast.",
    "Early mornings at the gym, healthy lifestyle focused.",
    "Want someone who shares similar wellness values."
  ],
  [
    "Journalist for the Chicago Tribune.",
    "Love discovering hidden gems and local stories.",
    "Seeking someone curious about the city and its culture."
  ],
  [
    "Architecture student at IIT.",
    "Fascinated by Chicago's architectural history.",
    "Looking for someone who appreciates good design."
  ],
  [
    "Social worker in community development.",
    "Passionate about making a positive impact.",
    "Want a roommate who's socially conscious and caring."
  ],
  [
    "Musician and music teacher.",
    "Play guitar and piano, love all genres of music.",
    "Need someone who's okay with occasional practice sessions."
  ],
  [
    "Data scientist at a healthcare company.",
    "Work from home most days, love coffee shops and libraries.",
    "Looking for someone who values quiet focus time."
  ],
  [
    "Chef at a trendy Wicker Park restaurant.",
    "Love experimenting with new recipes and ingredients.",
    "Want someone who enjoys good food and trying new cuisines."
  ],
  [
    "Veterinary student at University of Illinois.",
    "Animal lover with two cats (hypoallergenic!).",
    "Looking for a pet-friendly roommate who loves animals."
  ]
];

const habitsSummaries = [
  "Clean, organized, and respectful of shared spaces. I wash dishes right after use and keep common areas tidy.",
  "Generally quiet during weekdays, but enjoy having friends over on weekends. Always give advance notice.",
  "Early bird who's usually asleep by 10pm and up by 6am. Love morning coffee on the balcony.",
  "Social but understand the importance of personal space. Happy to chat or give privacy as needed.",
  "Night owl who does most creative work after 8pm. Use headphones and keep noise to a minimum.",
  "Busy schedule means I'm not home much during weekdays. Weekends are for relaxing and socializing.",
  "Love to cook and happy to share meals! Also enjoy keeping plants around the apartment.",
  "Work irregular hours but very considerate about noise. Prefer a calm, zen-like living environment.",
  "Extremely organized and like everything in its place. Happy to help maintain a structured home.",
  "Flexible and adaptable to different living styles. Good at compromising and communication.",
  "Love hosting game nights and small gatherings. Always check with roommates first though!",
  "Minimalist lifestyle - don't have much stuff and prefer uncluttered spaces.",
  "Health-conscious, keep the fridge stocked with fresh produce and healthy snacks.",
  "Creative mess maker when working on projects, but always clean up after myself.",
  "Regular cleaning schedule on Sundays, prefer to split household chores fairly.",
  "Love having fresh flowers and candles around. Keep shared spaces welcoming and cozy.",
  "Tech-savvy and happy to help with any home automation or internet setup.",
  "Occasional overnight shifts mean I sometimes come home at odd hours - very quiet though!",
  "Love decorating for holidays and seasons. Make the place feel festive and fun.",
  "Pet owner who's very responsible about cleaning and maintenance. Animals are well-trained."
];

const lookingForSummaries = [
  "Looking for a like-minded roommate who values cleanliness and good communication.",
  "Seeking someone responsible and easy-going who can become a good friend.",
  "Want a roommate who respects quiet hours but is also up for occasional hangouts.",
  "Looking for someone social and outgoing who enjoys the vibrant Chicago lifestyle.",
  "Need a roommate who's understanding of my work schedule and creative process.",
  "Seeking a studious, career-focused person who values a productive living environment.",
  "Want someone who's independent but also enjoys building a sense of community at home.",
  "Looking for a health-conscious roommate who appreciates work-life balance.",
  "Seeking an intellectually curious person who enjoys deep conversations.",
  "Want someone financially responsible who pays bills on time and respects shared expenses.",
  "Looking for a fellow foodie who enjoys cooking together and trying new restaurants.",
  "Need someone flexible and understanding of irregular work schedules.",
  "Seeking a fitness-minded person who might enjoy working out together occasionally.",
  "Want a roommate who's socially conscious and cares about community issues.",
  "Looking for someone who appreciates art, culture, and Chicago's creative scene.",
  "Seeking a compassionate person who shares similar values about kindness and respect.",
  "Want a music lover who might enjoy jamming together or going to concerts.",
  "Looking for a professional who maintains a good work-life balance.",
  "Seeking someone who loves animals and is comfortable with pets in the home.",
  "Want a roommate who's open to new experiences and exploring the city together."
];

// Helper function to generate random birthday (age between 22-35)
const generateRandomBirthday = (): Timestamp => {
  const today = new Date();
  const minAge = 22;
  const maxAge = 35;
  const age = Math.floor(Math.random() * (maxAge - minAge + 1)) + minAge;
  
  const birthYear = today.getFullYear() - age;
  const birthMonth = Math.floor(Math.random() * 12);
  const birthDay = Math.floor(Math.random() * 28) + 1; // Safe day range
  
  return Timestamp.fromDate(new Date(birthYear, birthMonth, birthDay));
};

// Helper function to calculate age from birthday
const calculateAge = (birthday: Timestamp): number => {
  const today = new Date();
  const birthDate = birthday.toDate();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  
  return age;
};

// Generate sample users
export const generateSampleUsers = (): Omit<User, 'id' | 'createdAt' | 'updatedAt'>[] => {
  const users: Omit<User, 'id' | 'createdAt' | 'updatedAt'>[] = [];
  
  // Better profile image sources with more realistic portraits
  const getProfileImageUrl = (index: number, gender: Gender): string => {
    // Using different services for variety and realism
    const sources = [
      // This Person Does Not Exist (AI-generated realistic faces)
      `https://picsum.photos/400/400?random=${index + 100}`,
      // Lorem Picsum with face-focused crops
      `https://picsum.photos/seed/${index + 200}/400/400`,
      // Unsplash with better search terms for portraits
      `https://images.unsplash.com/photo-${1500000000000 + index * 1000}?w=400&h=400&fit=crop&crop=face`,
      // Alternative Unsplash approach
      `https://picsum.photos/400/400?random=${index + 300}`
    ];
    
    return sources[index % sources.length];
  };
  
  // Generate 30 users for better testing variety
  for (let i = 0; i < 30; i++) {
    const birthday = generateRandomBirthday();
    const age = calculateAge(birthday);
    const genders: Gender[] = ['male', 'female', 'non-binary'];
    const profileTypes: ProfileType[] = ['looking_for_room', 'have_room'];
    const gender = genders[Math.floor(Math.random() * genders.length)];
    
    const user: Omit<User, 'id' | 'createdAt' | 'updatedAt'> = {
      email: `user${i + 1}@example.com`,
      name: names[i % names.length],
      birthday,
      age,
      gender,
      profileType: profileTypes[Math.floor(Math.random() * profileTypes.length)],
      profileImageUrl: getProfileImageUrl(i, gender),
      preferences: {
        cleanliness: (Math.floor(Math.random() * 5) + 1) as 1 | 2 | 3 | 4 | 5,
        noiseLevel: (Math.floor(Math.random() * 5) + 1) as 1 | 2 | 3 | 4 | 5,
        socialLevel: (Math.floor(Math.random() * 5) + 1) as 1 | 2 | 3 | 4 | 5,
        sleepSchedule: ['early_bird', 'night_owl', 'regular'][Math.floor(Math.random() * 3)] as 'early_bird' | 'night_owl' | 'regular',
        preferredRoommateGender: Math.random() > 0.5 ? 'any' : genders[Math.floor(Math.random() * genders.length)],
        preferredAgeRange: {
          min: Math.floor(Math.random() * 5) + 20, // 20-24
          max: Math.floor(Math.random() * 10) + 30 // 30-39
        },
        monthlyRentBudget: Math.floor(Math.random() * 1000) + 800 // $800-$1800
      },
      descriptions: descriptions[i % descriptions.length],
      habitsSummary: habitsSummaries[i % habitsSummaries.length],
      lookingForSummary: lookingForSummaries[i % lookingForSummaries.length]
    };
    
    users.push(user);
  }
  
  return users;
};

// Create users with sequential IDs and connect to existing storage folders
export const createUsersWithSequentialIds = async (): Promise<User[]> => {
  try {
    console.log('Step 1: Scanning existing Firebase Storage folders...');
    const storageUsersData = await scanAllStorageUsers();
    
    if (storageUsersData.length === 0) {
      throw new Error('No user folders found in Firebase Storage. Please upload images first.');
    }
    
    console.log(`Step 2: Found ${storageUsersData.length} users with images in storage`);
    console.log('Step 3: Creating users with sequential IDs and connecting to storage...');
    
    // Generate base user data
    const sampleUsers = generateSampleUsers();
    const uploadedUsers: User[] = [];
    
    // Clear existing users first
    const usersCollection = collection(db, 'users');
    const snapshot = await getDocs(usersCollection);
    const batch = writeBatch(db);
    
    snapshot.forEach((doc) => {
      batch.delete(doc.ref);
    });
    
    if (snapshot.size > 0) {
      await batch.commit();
      console.log(`Cleared ${snapshot.size} existing users`);
    }
    
    // Create new users with sequential IDs
    for (let i = 0; i < Math.min(storageUsersData.length, sampleUsers.length); i++) {
      const userData = sampleUsers[i];
      const storageData = storageUsersData[i];
      
      // Create sequential user ID with leading zeros
      const sequentialUserId = String(i + 1).padStart(5, '0'); // 00001, 00002, etc.
      const timestamp = Timestamp.now();
      
      // Extract userId from storage path if possible
      const storageUserIdMatch = storageData.userId.match(/\d+$/);
      const storageUserId = storageUserIdMatch ? String(parseInt(storageUserIdMatch[0])).padStart(5, '0') : sequentialUserId;
      
      // Build images object with storage URLs
      const imagesData: { profile: string[]; property?: string[] } = {
        profile: storageData.profileImages.length > 0 ? storageData.profileImages : [userData.profileImageUrl || '']
      };

      // Only add property field if there are actual property images
      if (storageData.propertyImages && storageData.propertyImages.length > 0) {
        imagesData.property = storageData.propertyImages;
      }

      const userWithSequentialId: User = {
        ...userData,
        id: sequentialUserId,
        userId: sequentialUserId, // Add explicit userId field matching the ID
        // Use first profile image as legacy profileImageUrl for backward compatibility
        profileImageUrl: storageData.profileImages[0] || userData.profileImageUrl,
        // Add structured images from storage
        images: imagesData,
        createdAt: timestamp,
        updatedAt: timestamp
      };
      
      const userRef = doc(db, 'users', sequentialUserId);
      await setDoc(userRef, userWithSequentialId);
      uploadedUsers.push(userWithSequentialId);
      
      console.log(`✅ Created user ${sequentialUserId}: ${userWithSequentialId.name}`);
      console.log(`   - Storage folder: ${storageData.userId}`);
      console.log(`   - Profile images: ${storageData.profileImages.length}`);
      console.log(`   - Property images: ${storageData.propertyImages.length}`);
    }
    
    console.log(`🎉 Successfully created ${uploadedUsers.length} users with sequential IDs and connected storage!`);
    return uploadedUsers;
  } catch (error) {
    console.error('Error creating users with sequential IDs:', error);
    throw error;
  }
};

// Upload users and images with proper sequential structure
export const uploadUsersWithSequentialIdsAndImages = async (): Promise<User[]> => {
  try {
    console.log('🚀 Starting complete setup with sequential user IDs and images...');
    
    // Step 1: Upload images to storage with proper structure
    console.log('Step 1: Uploading structured images to Firebase Storage...');
    await uploadSampleUsersImagesWithStructure();
    
    // Step 2: Create users with sequential IDs and connect to storage
    console.log('Step 2: Creating users with sequential IDs...');
    const users = await createUsersWithSequentialIds();
    
    console.log('✅ Complete setup finished successfully!');
    return users;
  } catch (error) {
    console.error('Error in complete setup:', error);
    throw error;
  }
};

// Upload sample users to Firebase with structured Firebase Storage images (ORIGINAL FUNCTION)
export const uploadSampleUsersWithStorageImages = async (): Promise<User[]> => {
  try {
    console.log('Step 1: Uploading structured images to Firebase Storage...');
    const usersImageData = await uploadSampleUsersImagesWithStructure();
    
    console.log('Step 2: Generating user profiles with structured storage URLs...');
    const sampleUsers = generateSampleUsers();
    const uploadedUsers: User[] = [];
    
    for (let i = 0; i < sampleUsers.length; i++) {
      const userData = sampleUsers[i];
      const userId = `sample_user_${i + 1}`;
      const timestamp = Timestamp.now();
      
      // Get the structured images for this user
      const userImages = usersImageData.find(data => data.userId === userId);
      
      // Build images object conditionally to avoid undefined fields
      const imagesData: { profile: string[]; property?: string[] } = {
        profile: userImages?.profileImages || [userData.profileImageUrl || '']
      };

      // Only add property field if there are actual property images
      if (userImages?.propertyImages && userImages.propertyImages.length > 0) {
        imagesData.property = userImages.propertyImages;
      }

      const userWithIds: User = {
        ...userData,
        id: userId,
        // Use first profile image as legacy profileImageUrl for backward compatibility
        profileImageUrl: userImages?.profileImages[0] || userData.profileImageUrl,
        // Add structured images
        images: imagesData,
        createdAt: timestamp,
        updatedAt: timestamp
      };
      
      const userRef = doc(db, 'users', userId);
      await setDoc(userRef, userWithIds);
      uploadedUsers.push(userWithIds);
      
      console.log(`Uploaded user: ${userWithIds.name} with ${userWithIds.images?.profile.length} profile images${userWithIds.images?.property ? ` and ${userWithIds.images.property.length} property images` : ''}`);
    }
    
    console.log(`Successfully uploaded ${uploadedUsers.length} sample users with structured Firebase Storage images`);
    return uploadedUsers;
  } catch (error) {
    console.error('Error uploading sample users with structured storage images:', error);
    // Fallback to regular upload if Firebase Storage fails
    console.log('Falling back to external image URLs...');
    return uploadSampleUsers();
  }
};

// Upload sample users to Firebase (ORIGINAL FUNCTION)
export const uploadSampleUsers = async (): Promise<User[]> => {
  try {
    const sampleUsers = generateSampleUsers();
    const uploadedUsers: User[] = [];
    
    for (let i = 0; i < sampleUsers.length; i++) {
      const userData = sampleUsers[i];
      const userId = `sample_user_${i + 1}`;
      const timestamp = Timestamp.now();
      
      const userWithIds: User = {
        ...userData,
        id: userId,
        createdAt: timestamp,
        updatedAt: timestamp
      };
      
      const userRef = doc(db, 'users', userId);
      await setDoc(userRef, userWithIds);
      uploadedUsers.push(userWithIds);
      
      console.log(`Uploaded user: ${userWithIds.name}`);
    }
    
    console.log(`Successfully uploaded ${uploadedUsers.length} sample users to Firebase`);
    return uploadedUsers;
  } catch (error) {
    console.error('Error uploading sample users:', error);
    throw error;
  }
};

// Get all users from Firebase
export const getAllUsers = async (): Promise<User[]> => {
  try {
    const usersCollection = collection(db, 'users');
    const snapshot = await getDocs(usersCollection);
    const users: User[] = [];
    
    snapshot.forEach((doc) => {
      users.push(doc.data() as User);
    });
    
    return users;
  } catch (error) {
    console.error('Error getting users:', error);
    throw error;
  }
};

// Delete sample users from Firebase (including their images)
export const clearSampleUsers = async (): Promise<void> => {
  try {
    console.log('Clearing sample users and their images...');
    
    // First, clear all structured images from Firebase Storage
    await clearAllSampleUserImages();
    
    // Then, clear user documents from Firestore
    const usersCollection = collection(db, 'users');
    const snapshot = await getDocs(usersCollection);
    
    const batch = writeBatch(db);
    let count = 0;
    
    snapshot.forEach((doc) => {
      if (doc.id.startsWith('sample_user_') || doc.id.match(/^\d{5}$/)) {
        batch.delete(doc.ref);
        count++;
      }
    });
    
    if (count > 0) {
      await batch.commit();
      console.log(`Successfully deleted ${count} sample users from Firebase`);
    } else {
      console.log('No sample users found to delete');
    }
    
    console.log('Sample user cleanup complete');
  } catch (error) {
    console.error('Error clearing sample users:', error);
    throw error;
  }
};