# Firebase Implementation in RoommateSwipe

## Overview

We've implemented Firebase Firestore integration to store user registration data from three different sign-up flows, each in its own collection:

1. **Looking for a Room** users - stored in `looking_for_room` collection
2. **Have a Room** users - stored in `have_room` collection
3. **Apartment Listing** companies - stored in `apartment_listings` collection

## Firebase Service Implementation

The `FirebaseService` class has been enhanced with methods to support these different user types:

- `saveLookingForRoomUser()` - Saves user data for people looking for a room
- `saveHaveRoomUser()` - Saves user data for people who have a room available
- `saveApartmentListing()` - Saves data for apartment companies listing properties

Each method:
- Accepts the appropriate registration data struct
- Creates a unique ID for the document
- Converts the struct data into a Firestore-friendly dictionary
- Stores the data in the appropriate collection
- Returns the generated ID and any error via completion handler

## UI Integration

All three sign-up views now include:

1. **Loading Indicators** - Overlays that show progress while saving to Firestore
2. **Success/Error Alerts** - Feedback to users on whether their data was saved successfully
3. **Firebase Integration** - Calls to the appropriate FirebaseService methods when registration is complete

## Data Structure

### Looking for a Room User
```
{
  "userId": String,
  "firstName": String,
  "lastName": String,
  "email": String,
  "dateOfBirth": String,
  "gender": String,
  "occupation": String,
  "bio": String,
  "cleanliness": Number,
  "noise": Number,
  "socialLevel": Number,
  "sleepSchedule": String,
  "drinking": Boolean,
  "smoking": Boolean,
  "pets": Boolean,
  "preferredGender": String,
  "ageRangeMin": Number,
  "ageRangeMax": Number,
  "rentRangeMin": Number,
  "rentRangeMax": Number,
  "moveInDate": String,
  "wakeUpTime": String,
  "bedTime": String,
  "workSchedule": String,
  "preferredNeighborhoods": Array<String>,
  "desiredAmenities": Array<String>,
  "roomPreference": String,
  "hasProfilePhoto": Boolean,
  "registrationDate": Timestamp,
  "userType": "lookingForRoom"
}
```

### Have a Room User
```
{
  "userId": String,
  "fullName": String,
  "email": String,
  "phoneNumber": String,
  "dateOfBirth": String,
  "city": String,
  "neighborhood": String,
  "moveInDate": String,
  "rentPrice": Number,
  "housingType": String,
  "isFurnished": Boolean,
  "genderPreference": String,
  "smokingPreference": Boolean,
  "petPreference": Boolean,
  "lifestylePreference": String,
  "cleanlinessLevel": Number,
  "bio": String,
  "interests": Array<String>,
  "hasProfilePhoto": Boolean,
  "hasRoomPhotos": Boolean,
  "roomPhotoCount": Number,
  "registrationDate": Timestamp,
  "userType": "haveRoom"
}
```

### Apartment Listing
```
{
  "listingId": String,
  "companyName": String,
  "contactPersonName": String,
  "email": String,
  "contactPhoneNumber": String,
  "city": String,
  "neighborhood": String,
  "apartmentName": String,
  "address": String,
  "numberOfAvailableRooms": Number,
  "availabilityDate": String,
  "minPrice": Number,
  "maxPrice": Number,
  "housingType": String,
  "amenities": Array<String>,
  "isPetFriendly": Boolean,
  "isSmokingAllowed": Boolean,
  "propertyDescription": String,
  "hasPropertyPhotos": Boolean,
  "propertyPhotoCount": Number,
  "listingDate": Timestamp,
  "userType": "apartmentListing"
}
```

## Notes on Image Handling

Currently, we're only recording whether the user has provided images (profile photos, property photos, etc.) without actually uploading them to Firebase Storage. In a full implementation, we would:

1. Upload images to Firebase Storage
2. Store the resulting URLs in the Firestore documents
3. Use the URLs to download and display images in the app

## Testing Firebase Integration

The app includes a test function in `FirebaseService`:

```swift
func testFirebaseConnection(completion: @escaping (Bool, Error?) -> Void)
```

This writes a simple test document to the "test" collection to verify that Firebase is properly connected.

## Next Steps

1. **Firebase Storage Integration** - Add proper storage for user images
2. **User Authentication** - Implement Firebase Authentication for secure login
3. **Realtime Data** - Use Firestore listeners for realtime updates
4. **Query Implementation** - Add methods to query users/listings based on preferences
5. **Data Validation** - Add client-side validation before saving to Firebase 