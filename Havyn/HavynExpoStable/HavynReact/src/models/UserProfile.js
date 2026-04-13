export default class UserProfile {
  constructor(data = {}) {
    this.id = data.id || '';
    this.email = data.email || '';
    this.fullName = data.fullName || '';
    this.birthdate = data.birthdate || null;
    this.gender = data.gender || '';
    this.phoneNumber = data.phoneNumber || '';
    this.bio = data.bio || '';
    this.profileImageUrl = data.profileImageUrl || '';
    this.additionalPhotos = data.additionalPhotos || [];
    this.occupation = data.occupation || '';
    this.school = data.school || '';
    
    // Lifestyle preferences
    this.cleanliness = data.cleanliness || 3; // Scale of 1-5
    this.noise = data.noise || 3; // Scale of 1-5
    this.guestsFrequency = data.guestsFrequency || 'Occasionally';
    this.wakeTime = data.wakeTime || '';
    this.sleepTime = data.sleepTime || '';
    this.smoking = data.smoking || false;
    this.drinking = data.drinking || 'Occasionally';
    this.pets = data.pets || false;
    this.dietaryRestrictions = data.dietaryRestrictions || [];
    
    // Roommate preferences
    this.preferredGender = data.preferredGender || 'Any';
    this.ageRangeMin = data.ageRangeMin || 18;
    this.ageRangeMax = data.ageRangeMax || 99;
    this.budget = data.budget || { min: 0, max: 5000 };
    
    // Location preferences
    this.desiredLocations = data.desiredLocations || [];
    this.currentLocation = data.currentLocation || null;
    this.moveInDate = data.moveInDate || null;
    
    // App data
    this.accountType = data.accountType || 'roommate'; // 'roommate', 'has_room', 'property_manager'
    this.likes = data.likes || [];
    this.dislikes = data.dislikes || [];
    this.matches = data.matches || [];
    this.conversations = data.conversations || {};
    this.createdAt = data.createdAt || new Date();
    this.lastActive = data.lastActive || new Date();
    this.isVerified = data.isVerified || false;
    
    // Property info (for has_room)
    this.property = data.property || null;
  }
  
  toJSON() {
    return {
      id: this.id,
      email: this.email,
      fullName: this.fullName,
      birthdate: this.birthdate,
      gender: this.gender,
      phoneNumber: this.phoneNumber,
      bio: this.bio,
      profileImageUrl: this.profileImageUrl,
      additionalPhotos: this.additionalPhotos,
      occupation: this.occupation,
      school: this.school,
      
      cleanliness: this.cleanliness,
      noise: this.noise,
      guestsFrequency: this.guestsFrequency,
      wakeTime: this.wakeTime,
      sleepTime: this.sleepTime,
      smoking: this.smoking,
      drinking: this.drinking,
      pets: this.pets,
      dietaryRestrictions: this.dietaryRestrictions,
      
      preferredGender: this.preferredGender,
      ageRangeMin: this.ageRangeMin,
      ageRangeMax: this.ageRangeMax,
      budget: this.budget,
      
      desiredLocations: this.desiredLocations,
      currentLocation: this.currentLocation,
      moveInDate: this.moveInDate,
      
      accountType: this.accountType,
      likes: this.likes,
      dislikes: this.dislikes,
      matches: this.matches,
      conversations: this.conversations,
      createdAt: this.createdAt,
      lastActive: this.lastActive,
      isVerified: this.isVerified,
      
      property: this.property,
    };
  }
} 