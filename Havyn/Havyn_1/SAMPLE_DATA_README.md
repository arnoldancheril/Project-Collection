# Sample Data Setup Guide

## Overview
This guide explains how to populate your Havyn app with sample user profiles for development and testing purposes.

## Features Added

### 1. Sample Data Service (`src/services/sampleDataService.ts`)
- **`generateSampleUsers()`**: Creates 30 realistic user profiles with varied data
- **`uploadSampleUsers()`**: Uploads sample profiles to Firebase Firestore
- **`getAllUsers()`**: Retrieves all user profiles from Firebase
- **`clearSampleUsers()`**: Removes all sample users from the database

### 2. ProfileCard Component (`src/components/ProfileCard.tsx`)
- Displays user profiles in a card format
- Shows profile image, name, age, profile type
- Displays user descriptions, habits, and preferences
- Responsive design with rating stars for preferences

### 3. Updated Home Screen (`app/(tabs)/index.tsx`)
- Lists all user profiles from Firebase
- "Upload 30 Sample Profiles" button when no profiles exist
- Pull-to-refresh functionality
- Replace/Clear sample data buttons for testing
- Shows count of sample vs real profiles

## How to Use

### 1. Running the App
```bash
npm start
```

### 2. Adding Sample Data
1. Open the app and navigate to the Home tab
2. If no profiles exist, tap "Upload 30 Sample Profiles"
3. Confirm the action in the alert dialog
4. Wait for the success message
5. Profiles will automatically load and display

### 3. Managing Sample Data
- **Replace Samples**: Updates existing sample data with 30 new profiles
- **Clear Samples**: Removes all sample profiles from the database
- **Pull to Refresh**: Reloads data from Firebase

## Sample Data Structure

Each sample user includes:
- **Personal Info**: Name, age, gender, email
- **Profile Type**: Looking for room or has room available
- **Preferences**: Cleanliness, noise level, social level, sleep schedule
- **Descriptions**: Array of 3 personality/lifestyle descriptions
- **Habits Summary**: Living habits and routines
- **Looking For Summary**: What they want in a roommate
- **Budget**: Monthly rent budget (if looking for room)

## Data Variety

The expanded sample data includes:

### **30 Diverse Profiles** with:
- **Ages**: 22-35 years old
- **Genders**: Male, female, non-binary
- **Profile Types**: Looking for room, have room available

### **Professional Backgrounds**:
- **Students**: Graduate students (Economics, Neuroscience, Law, Architecture, Veterinary)
- **Healthcare**: Medical residents, nurses, social workers
- **Technology**: Software engineers, data scientists
- **Creative**: Artists, musicians, journalists, chefs
- **Education**: Teachers, music instructors
- **Business**: Financial analysts, marketing coordinators
- **Service**: Personal trainers, bartenders

### **Lifestyle Variety**:
- Early birds vs night owls
- Social butterflies vs quiet types
- Health-conscious individuals
- Pet owners and animal lovers
- Tech-savvy professionals
- Creative types with flexible schedules
- Fitness enthusiasts
- Foodies and cooking enthusiasts

### **Budget Range**: $800-$1800/month rent

## Firebase Integration

### Collections Used
- `users`: Stores all user profiles
- Sample users have IDs like `sample_user_1`, `sample_user_2`, etc. (up to `sample_user_30`)

### Data Validation
- All data follows the `User` interface defined in `src/models/User.ts`
- Timestamps are properly handled with Firebase Timestamp
- Age is calculated from birthday
- All required fields are populated

## Development Notes

### Adding More Sample Data
To expand beyond 30 profiles, modify the arrays in `sampleDataService.ts`:
- `names`: Add more diverse names
- `descriptions`: Add more personality descriptions (each should be an array of 3 strings)
- `habitsSummaries`: Add more living habit descriptions
- `lookingForSummaries`: Add more roommate preference descriptions
- Update the loop in `generateSampleUsers()` to create more profiles

### Customizing Profile Display
Modify `ProfileCard.tsx` to:
- Change the visual design
- Add/remove displayed fields
- Modify the rating system
- Adjust card layout

### Testing Backend Connection
The sample data functionality serves as a test for:
- Firebase Firestore read/write operations
- Data serialization/deserialization
- Real-time data updates
- Error handling
- User interface responsiveness
- Batch operations for clearing data

## Sample Profile Categories

### **Academic Professionals** (8 profiles)
- Graduate students in various fields
- Medical and veterinary students
- Research-focused individuals

### **Tech Workers** (6 profiles)
- Software engineers
- Data scientists
- Remote workers

### **Healthcare Workers** (4 profiles)
- Medical residents
- Nurses
- Social workers

### **Creative Professionals** (6 profiles)
- Artists and designers
- Musicians and teachers
- Journalists and writers
- Chefs and culinary professionals

### **Service & Business** (6 profiles)
- Personal trainers
- Bartenders and hospitality
- Financial analysts
- Marketing professionals

## Troubleshooting

### Common Issues
1. **Firebase Connection**: Ensure `firebaseConfig.js` is properly configured
2. **TypeScript Errors**: Check that all imports are correct
3. **Data Not Loading**: Check console for Firebase permissions errors
4. **Images Not Loading**: Profile images use placeholder URLs that may not always load
5. **Duplicate Data**: Use "Replace Samples" to clear and reload fresh data

### Debug Commands
```bash
# Check TypeScript errors
npx tsc --noEmit

# Clear cache and restart
npx expo start --clear

# Check Firebase rules in Firebase Console
```

### Testing Workflow
1. **Upload Samples**: Test Firebase write operations
2. **View Profiles**: Test UI rendering and data display
3. **Replace Samples**: Test batch delete and write operations
4. **Clear Samples**: Test batch delete operations
5. **Pull to Refresh**: Test Firebase read operations

## Next Steps

With 30 diverse sample profiles, you can now:
1. **Test UI Performance**: See how the app handles larger datasets
2. **Test Filtering/Search**: Implement and test search functionality
3. **Test Matching Algorithms**: Build roommate matching features
4. **Implement User Authentication**: Add real user registration
5. **Add Photo Upload**: Replace placeholder images with real uploads
6. **Create Messaging**: Build chat functionality between matched users
7. **Add Geolocation**: Implement location-based features

The expanded sample data provides a robust foundation for continued development and testing of the Havyn roommate finder app, offering sufficient variety to test various app features and user scenarios. 