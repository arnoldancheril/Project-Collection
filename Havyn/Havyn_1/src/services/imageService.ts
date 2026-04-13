import { ref, uploadBytes, getDownloadURL, deleteObject, listAll } from 'firebase/storage';
import { storage } from '../../firebaseConfig';

// Upload a single image to Firebase Storage with user-specific folder structure
export const uploadImageToStorage = async (
  imageUri: string, 
  userId: string,
  imageType: 'profile' | 'property',
  fileName: string
): Promise<string> => {
  try {
    // Convert URI to blob for upload
    const response = await fetch(imageUri);
    const blob = await response.blob();
    
    // Format userId to ensure it uses the sequential format (00001, 00002, etc.)
    // If userId is already in proper format, this won't change it
    const formattedUserId = userId.startsWith('sample_user_')
      ? String(userId.split('_')[2]).padStart(5, '0')
      : userId.match(/^\d+$/)
        ? String(parseInt(userId)).padStart(5, '0')
        : userId;
    
    // Create user-specific storage reference: users/{userId}/{imageType}/{fileName}
    const imagePath = `users/${formattedUserId}/${imageType}/${fileName}`;
    const imageRef = ref(storage, imagePath);
    
    // Upload the blob
    await uploadBytes(imageRef, blob);
    
    // Get the download URL
    const downloadURL = await getDownloadURL(imageRef);
    
    console.log(`Image uploaded successfully to ${imagePath}: ${downloadURL}`);
    return downloadURL;
  } catch (error) {
    console.error('Error uploading image:', error);
    throw error;
  }
};

// Upload multiple profile images for a user
export const uploadUserProfileImages = async (
  userId: string,
  imageUrls: string[],
  maxImages: number = 5
): Promise<string[]> => {
  try {
    const uploadedUrls: string[] = [];
    const imagesToUpload = imageUrls.slice(0, maxImages);
    
    console.log(`Uploading ${imagesToUpload.length} profile images for user ${userId}...`);
    
    for (let i = 0; i < imagesToUpload.length; i++) {
      try {
        const externalUrl = imagesToUpload[i];
        const fileName = `profile_${i + 1}.jpg`;
        
        const firebaseUrl = await uploadImageToStorage(externalUrl, userId, 'profile', fileName);
        uploadedUrls.push(firebaseUrl);
        
        // Small delay to avoid overwhelming the service
        await new Promise(resolve => setTimeout(resolve, 200));
      } catch (error) {
        console.error(`Failed to upload profile image ${i + 1} for user ${userId}:`, error);
        // Use fallback URL if upload fails
        uploadedUrls.push(`https://picsum.photos/400/400?random=${userId}_${i}`);
      }
    }
    
    return uploadedUrls;
  } catch (error) {
    console.error('Error uploading user profile images:', error);
    throw error;
  }
};

// Upload property images for a user
export const uploadUserPropertyImages = async (
  userId: string,
  imageUrls: string[]
): Promise<string[]> => {
  try {
    const uploadedUrls: string[] = [];
    
    console.log(`Uploading ${imageUrls.length} property images for user ${userId}...`);
    
    const propertyImageTypes = ['room_main', 'room_detail', 'common_area', 'kitchen', 'bathroom'];
    
    for (let i = 0; i < Math.min(imageUrls.length, propertyImageTypes.length); i++) {
      try {
        const externalUrl = imageUrls[i];
        const fileName = `${propertyImageTypes[i]}.jpg`;
        
        const firebaseUrl = await uploadImageToStorage(externalUrl, userId, 'property', fileName);
        uploadedUrls.push(firebaseUrl);
        
        await new Promise(resolve => setTimeout(resolve, 200));
      } catch (error) {
        console.error(`Failed to upload property image ${i + 1} for user ${userId}:`, error);
        uploadedUrls.push(`https://picsum.photos/600/400?random=property_${userId}_${i}`);
      }
    }
    
    return uploadedUrls;
  } catch (error) {
    console.error('Error uploading user property images:', error);
    throw error;
  }
};

// Download sample images and upload with organized structure
export const uploadSampleUsersImagesWithStructure = async (): Promise<{userId: string, profileImages: string[], propertyImages: string[]}[]> => {
  try {
    const usersData: {userId: string, profileImages: string[], propertyImages: string[]}[] = [];
    
    console.log('Starting structured upload of sample images to Firebase Storage...');
    
    // Generate images for 30 users
    for (let i = 0; i < 30; i++) {
      const userId = `sample_user_${i + 1}`;
      
      // Generate external URLs for profile images (2-4 per user)
      const profileImageCount = Math.floor(Math.random() * 3) + 2; // 2-4 images
      const profileExternalUrls = Array.from({length: profileImageCount}, (_, idx) => 
        `https://picsum.photos/400/400?random=${1000 + i * 10 + idx}`
      );
      
      // Generate external URLs for property images (only for users with rooms)
      const hasProperty = Math.random() > 0.6; // 40% have property images
      const propertyExternalUrls = hasProperty ? [
        `https://picsum.photos/600/400?random=${2000 + i * 10}`, // room main
        `https://picsum.photos/600/400?random=${2000 + i * 10 + 1}`, // room detail
        `https://picsum.photos/600/400?random=${2000 + i * 10 + 2}`, // common area
        `https://picsum.photos/600/400?random=${2000 + i * 10 + 3}`, // kitchen
      ] : [];
      
      console.log(`Processing user ${i + 1}/30: ${userId} (${profileImageCount} profile, ${propertyExternalUrls.length} property)`);
      
      // Upload profile images
      const profileImages = await uploadUserProfileImages(userId, profileExternalUrls);
      
      // Upload property images if applicable
      const propertyImages = propertyExternalUrls.length > 0 
        ? await uploadUserPropertyImages(userId, propertyExternalUrls)
        : [];
      
      usersData.push({
        userId,
        profileImages,
        propertyImages
      });
      
      // Longer delay between users to avoid overwhelming Firebase
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log(`Successfully uploaded structured images for ${usersData.length} users`);
    return usersData;
  } catch (error) {
    console.error('Error uploading structured sample images:', error);
    throw error;
  }
};

// Get all images for a specific user
export const getUserImages = async (userId: string): Promise<{profile: string[], property: string[]}> => {
  try {
    const profileRef = ref(storage, `users/${userId}/profile`);
    const propertyRef = ref(storage, `users/${userId}/property`);
    
    const [profileResult, propertyResult] = await Promise.allSettled([
      listAll(profileRef),
      listAll(propertyRef)
    ]);
    
    const profileUrls: string[] = [];
    const propertyUrls: string[] = [];
    
    // Get profile image URLs
    if (profileResult.status === 'fulfilled') {
      for (const itemRef of profileResult.value.items) {
        const url = await getDownloadURL(itemRef);
        profileUrls.push(url);
      }
    }
    
    // Get property image URLs
    if (propertyResult.status === 'fulfilled') {
      for (const itemRef of propertyResult.value.items) {
        const url = await getDownloadURL(itemRef);
        propertyUrls.push(url);
      }
    }
    
    return { profile: profileUrls, property: propertyUrls };
  } catch (error) {
    console.error(`Error getting images for user ${userId}:`, error);
    return { profile: [], property: [] };
  }
};

// Delete all images for a specific user
export const deleteUserImages = async (userId: string): Promise<void> => {
  try {
    const userRef = ref(storage, `users/${userId}`);
    const result = await listAll(userRef);
    
    // Delete all files in user's folder recursively
    const deletePromises: Promise<void>[] = [];
    
    // Delete files in subfolders
    for (const folderRef of result.prefixes) {
      const folderResult = await listAll(folderRef);
      folderResult.items.forEach(itemRef => {
        deletePromises.push(deleteObject(itemRef));
      });
    }
    
    // Delete any files directly in user folder
    result.items.forEach(itemRef => {
      deletePromises.push(deleteObject(itemRef));
    });
    
    await Promise.all(deletePromises);
    console.log(`Deleted all images for user ${userId}`);
  } catch (error) {
    console.error(`Error deleting images for user ${userId}:`, error);
    throw error;
  }
};

// Clear all sample user images (organized structure)
export const clearAllSampleUserImages = async (): Promise<void> => {
  try {
    console.log('Clearing all sample user images...');
    
    // Scan all existing storage folders
    const usersRef = ref(storage, 'users');
    const result = await listAll(usersRef);
    
    let clearedUsers = 0;
    
    // Delete images for all user types
    for (const userFolderRef of result.prefixes) {
      const userId = userFolderRef.name;
      
      // Delete both legacy sample_user_x IDs and sequential IDs (00001, 00002, etc.)
      if (userId.startsWith('sample_user_') || /^\d+$/.test(userId)) {
        try {
          await deleteUserImages(userId);
          clearedUsers++;
          console.log(`Deleted images for user: ${userId}`);
        } catch (error) {
          console.warn(`Could not delete images for ${userId}:`, error);
        }
      }
    }
    
    console.log(`Successfully cleared images for ${clearedUsers} users`);
  } catch (error) {
    console.error('Error clearing sample user images:', error);
    throw error;
  }
};

// Get storage statistics
export const getStorageStats = async (): Promise<{totalUsers: number, totalImages: number}> => {
  try {
    const usersRef = ref(storage, 'users');
    const result = await listAll(usersRef);
    
    let totalImages = 0;
    
    for (const userFolderRef of result.prefixes) {
      const userResult = await listAll(userFolderRef);
      
      // Count images in profile and property folders
      for (const subFolderRef of userResult.prefixes) {
        const subFolderResult = await listAll(subFolderRef);
        totalImages += subFolderResult.items.length;
      }
      
      // Count any direct files in user folder
      totalImages += userResult.items.length;
    }
    
    return {
      totalUsers: result.prefixes.length,
      totalImages
    };
  } catch (error) {
    console.error('Error getting storage stats:', error);
    return { totalUsers: 0, totalImages: 0 };
  }
};

// Scan all existing user folders in Storage and get their images
export const scanAllStorageUsers = async (): Promise<{userId: string, profileImages: string[], propertyImages: string[]}[]> => {
  try {
    console.log('Scanning all existing user folders in Firebase Storage...');
    
    const usersRef = ref(storage, 'users');
    const result = await listAll(usersRef);
    
    const usersData: {userId: string, profileImages: string[], propertyImages: string[]}[] = [];
    
    for (const userFolderRef of result.prefixes) {
      const userId = userFolderRef.name; // This gets the folder name (e.g., "sample_user_1" or "00001")
      console.log(`Scanning user folder: ${userId}`);
      
      // Format userId for consistent recognition - normalize sequential IDs
      let normalizedUserId = userId;
      
      // If it's a sequential ID (just digits), ensure it's properly formatted (00001, 00002, etc.)
      if (/^\d+$/.test(userId)) {
        normalizedUserId = String(parseInt(userId)).padStart(5, '0');
        console.log(`Normalized sequential user ID: ${userId} → ${normalizedUserId}`);
      }
      
      const userImages = await getUserImages(userId);
      
      usersData.push({
        userId: normalizedUserId, // Use the normalized ID for consistent matching
        profileImages: userImages.profile,
        propertyImages: userImages.property
      });
      
      console.log(`Found ${userImages.profile.length} profile images and ${userImages.property.length} property images for ${normalizedUserId}`);
    }
    
    // Sort users by ID for better organization and sequential display
    usersData.sort((a, b) => {
      // Extract numeric value for sorting
      const aNum = parseInt(a.userId.replace(/\D/g, '') || '0');
      const bNum = parseInt(b.userId.replace(/\D/g, '') || '0');
      return aNum - bNum;
    });
    
    console.log(`Scanned ${usersData.length} user folders in Storage`);
    return usersData;
  } catch (error) {
    console.error('Error scanning storage users:', error);
    throw error;
  }
};

// Legacy functions for backward compatibility
export const uploadSampleImagesToStorage = uploadSampleUsersImagesWithStructure;
export const getStorageImageUrls = async (): Promise<string[]> => {
  const stats = await getStorageStats();
  console.log(`Storage contains ${stats.totalUsers} users with ${stats.totalImages} total images`);
  return [`Found ${stats.totalImages} images across ${stats.totalUsers} users`];
};
export const clearStorageImages = clearAllSampleUserImages; 