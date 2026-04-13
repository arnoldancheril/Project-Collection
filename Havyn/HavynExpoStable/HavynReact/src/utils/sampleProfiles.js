// Sample profiles for the Havyn app
// Based on data from RoommateViewModel.swift

export const SAMPLE_PROFILES = [
  {
    id: '1',
    firstName: 'Elsa',
    lastName: 'Frozen',
    age: 21,
    gender: 'Female',
    occupation: 'Occasionally freezes the living room, loves to sing.',
    bio: 'Cool and collected roommate with a talent for interior decorating. I keep things neat and tidy.',
    images: [require('../../ProfileImages/profile1/image1.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'Love them',
      drinking: 'Social drinker',
      cleanliness: 'Very clean',
      guests: 'Occasionally',
      schedule: 'Early bird'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'River North'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 30)), // 30 days from now
    budget: {
      min: 800,
      max: 1200
    },
    property: {
      type: 'Apartment',
      rooms: 2,
      bathrooms: 1,
      rent: '$1200/month',
      address: '420 N State St, Chicago, IL 60654',
      amenities: ['Private balcony', 'Modern decor', 'Fireplace'],
      coordinate: {
        latitude: 41.8962,
        longitude: -87.6362
      }
    },
    habits: 'Early riser, clean and organized, quiet after midnight.',
    lookingFor: 'Someone who appreciates a well-maintained living space.',
    verified: true
  },
  {
    id: '2',
    firstName: 'Batman',
    lastName: 'Wayne',
    age: 30,
    gender: 'Male',
    occupation: 'Tech professional',
    bio: 'Tech professional, enjoys city views and modern amenities. I work long hours but keep a tidy space.',
    images: [require('../../ProfileImages/profile2/image1.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'No pets',
      drinking: 'Rarely',
      cleanliness: 'Very clean',
      guests: 'Rarely',
      schedule: 'Early bird'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'Loop'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 14)), // 14 days from now
    budget: {
      min: 4000,
      max: 6000
    },
    property: {
      type: 'Condo',
      rooms: 3,
      bathrooms: 2,
      rent: '$4500/month',
      address: '300 E Randolph St, Chicago, IL 60601',
      amenities: ['High-rise views', 'Gym', 'Doorman'],
      coordinate: {
        latitude: 41.8786,
        longitude: -87.6251
      }
    },
    habits: 'Work from home, gym enthusiast, minimal cooking.',
    lookingFor: 'Professional roommate with similar schedule.',
    verified: true
  },
  {
    id: '3',
    firstName: 'Homer',
    lastName: 'Simpson',
    age: 39,
    gender: 'Male',
    occupation: 'Nuclear Safety Inspector',
    bio: 'Loves donuts, craft beer enthusiast. Looking for a social roommate who enjoys hanging out.',
    images: [require('../../ProfileImages/profile3/image1.jpg')],
    lifestylePreferences: {
      smoking: 'Outdoors',
      pets: 'Love them',
      drinking: 'Social drinker',
      cleanliness: 'Moderately clean',
      guests: 'Frequently',
      schedule: 'Regular hours'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'Wicker Park'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 7)), // 7 days from now
    budget: {
      min: 800,
      max: 1200
    },
    property: {
      type: 'Apartment',
      rooms: 4,
      bathrooms: 2,
      rent: '$1100/month',
      address: '1550 N Milwaukee Ave, Chicago, IL 60622',
      amenities: ['Rooftop deck', 'Garage parking', 'In-unit laundry'],
      coordinate: {
        latitude: 41.9088,
        longitude: -87.6796
      }
    },
    habits: 'Social, loves hosting game nights, casual living style.',
    lookingFor: 'Someone who enjoys a laid-back atmosphere.',
    verified: true
  },
  {
    id: '4',
    firstName: 'Frodo',
    lastName: 'Baggins',
    age: 28,
    gender: 'Male',
    occupation: 'Artist and musician',
    bio: 'Artist and musician, looking for creative roommates. I have a small studio space I use for practice.',
    images: [require('../../ProfileImages/profile4/image1.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'No pets',
      drinking: 'Occasionally',
      cleanliness: 'Moderately clean',
      guests: 'Occasionally',
      schedule: 'Night owl'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'Logan Square'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 14)), // 14 days from now
    budget: {
      min: 600,
      max: 800
    },
    property: {
      type: 'Apartment',
      rooms: 2,
      bathrooms: 1,
      rent: '$750/month',
      address: '2500 N Milwaukee Ave, Chicago, IL 60647',
      amenities: ['Music room', 'Art studio space', 'Garden'],
      coordinate: {
        latitude: 41.9231,
        longitude: -87.7093
      }
    },
    habits: 'Night owl, creative projects, occasional band practice.',
    lookingFor: 'Fellow artist or musician who appreciates creative energy.',
    verified: true
  },
  {
    id: '5',
    firstName: 'Buzz',
    lastName: 'Lightyear',
    age: 25,
    gender: 'Female',
    occupation: 'Yoga instructor',
    bio: 'Yoga instructor, plant enthusiast. Living a mindful lifestyle and hoping to find like-minded roommates.',
    images: [require('../../assets/person-placeholder.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'Love them',
      drinking: 'Occasionally',
      cleanliness: 'Very clean',
      guests: 'Occasionally',
      schedule: 'Early bird'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'Lincoln Park'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 30)), // 30 days from now
    budget: {
      min: 1000,
      max: 1500
    },
    property: {
      type: 'Apartment',
      rooms: 2,
      bathrooms: 2,
      rent: '$1300/month',
      address: '2000 N Lincoln Park W, Chicago, IL 60614',
      amenities: ['Yoga space', 'Balcony garden', 'Natural light'],
      coordinate: {
        latitude: 41.9214,
        longitude: -87.6513
      }
    },
    habits: 'Morning meditation, plant care, healthy cooking.',
    lookingFor: 'Health-conscious roommate who enjoys a peaceful home.',
    verified: true
  },
  {
    id: '6',
    firstName: 'Shrek',
    lastName: 'Ogre',
    age: 32,
    gender: 'Male',
    occupation: 'Tech startup founder',
    bio: 'Tech startup founder, coffee addict, fitness enthusiast. Looking for serious-minded roommates.',
    images: [require('../../assets/person-placeholder.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'No pets',
      drinking: 'Occasionally',
      cleanliness: 'Very clean',
      guests: 'Occasionally',
      schedule: 'Early bird'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'West Loop'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 21)), // 21 days from now
    budget: {
      min: 2000,
      max: 3000
    },
    property: {
      type: 'Loft',
      rooms: 3,
      bathrooms: 2,
      rent: '$2500/month',
      address: '1000 W Randolph St, Chicago, IL 60607',
      amenities: ['Home office', 'Smart home features', 'Fitness room'],
      coordinate: {
        latitude: 41.8857,
        longitude: -87.6478
      }
    },
    habits: 'Early morning workouts, works from home, loves cooking.',
    lookingFor: 'Career-focused professional who values fitness and healthy living.',
    verified: true
  },
  {
    id: '7',
    firstName: 'Scooby',
    lastName: 'Doo',
    age: 27,
    gender: 'Female',
    occupation: 'Freelance photographer',
    bio: 'Freelance photographer, world traveler, foodie. I travel frequently but keep a neat space when home.',
    images: [require('../../assets/person-placeholder.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'Love them',
      drinking: 'Social drinker',
      cleanliness: 'Very clean',
      guests: 'Occasionally',
      schedule: 'Flexible'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'Bucktown'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 45)), // 45 days from now
    budget: {
      min: 1200,
      max: 1800
    },
    property: {
      type: 'Apartment',
      rooms: 2,
      bathrooms: 1,
      rent: '$1600/month',
      address: '1800 N Damen Ave, Chicago, IL 60647',
      amenities: ['Photography studio', 'Rooftop access', 'Vintage charm'],
      coordinate: {
        latitude: 41.9169,
        longitude: -87.6762
      }
    },
    habits: 'Frequent traveler, home photography studio, loves hosting dinner parties.',
    lookingFor: 'Creative individual who appreciates art and good food.',
    verified: true
  },
  {
    id: '8',
    firstName: 'Aladdin',
    lastName: 'Jasmine',
    age: 29,
    gender: 'Male',
    occupation: 'Jazz musician',
    bio: 'Jazz musician, vinyl collector, culinary student. Looking for music-loving roommates.',
    images: [require('../../assets/person-placeholder.jpg')],
    lifestylePreferences: {
      smoking: 'No',
      pets: 'No pets',
      drinking: 'Social drinker',
      cleanliness: 'Moderately clean',
      guests: 'Occasionally',
      schedule: 'Night owl'
    },
    location: {
      city: 'Chicago',
      state: 'IL',
      neighborhood: 'Hyde Park'
    },
    moveInDate: new Date(new Date().setDate(new Date().getDate() + 15)), // 15 days from now
    budget: {
      min: 900,
      max: 1400
    },
    property: {
      type: 'Apartment',
      rooms: 3,
      bathrooms: 2,
      rent: '$1200/month',
      address: '5200 S Lake Shore Dr, Chicago, IL 60615',
      amenities: ['Music room', 'Chef\'s kitchen', 'Record collection space'],
      coordinate: {
        latitude: 41.7943,
        longitude: -87.5917
      }
    },
    habits: 'Late night practice sessions, cooking experiments, record collecting.',
    lookingFor: 'Music lover who enjoys good food and late nights.',
    verified: true
  }
];

export default SAMPLE_PROFILES; 