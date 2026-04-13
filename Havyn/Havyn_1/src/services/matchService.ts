import { User } from '../models/User';
import { Match } from '../models/Match';
import { Timestamp } from 'firebase/firestore';
import { getAllUsers } from './sampleDataService';

// Current user ID for testing - in a real app this would come from authentication
const CURRENT_USER_ID = '00001';

/**
 * Generate random matches for the current user
 * This is for development/testing purposes only
 */
export const generateRandomMatches = async (count: number = 5): Promise<Match[]> => {
  try {
    // Get all users
    const allUsers = await getAllUsers();
    
    // Filter out current user
    const otherUsers = allUsers.filter(user => user.userId !== CURRENT_USER_ID);
    
    // Randomly select users to match with
    const maxMatches = Math.min(count, otherUsers.length);
    const shuffled = [...otherUsers].sort(() => 0.5 - Math.random());
    const randomUsers = shuffled.slice(0, maxMatches);
    
    // Create match objects
    const matches: Match[] = randomUsers.map(user => ({
      id: `${CURRENT_USER_ID}_${user.userId || user.id}`,
      initiatorId: CURRENT_USER_ID,
      recipientId: user.userId || user.id,
      status: 'accepted',
      createdAt: Timestamp.now(),
      updatedAt: Timestamp.now(),
      hasUnreadMessages: Math.random() > 0.5 // Randomly set unread status
    }));
    
    return matches;
  } catch (error) {
    console.error('Error generating random matches:', error);
    throw error;
  }
};

/**
 * Get user objects for matches
 */
export const getMatchedUsers = async (matches: Match[]): Promise<User[]> => {
  try {
    // Get all users
    const allUsers = await getAllUsers();
    
    // Filter to get only matched users
    const matchedUserIds = matches.map(match => 
      match.initiatorId === CURRENT_USER_ID ? match.recipientId : match.initiatorId
    );
    
    const matchedUsers = allUsers.filter(user => 
      matchedUserIds.includes(user.userId || user.id)
    );
    
    return matchedUsers;
  } catch (error) {
    console.error('Error getting matched users:', error);
    throw error;
  }
};

/**
 * Simulate a random match for a user
 * Returns true if it's a match, false otherwise
 */
export const simulateMatchDecision = (matchProbability: number = 0.7): boolean => {
  return Math.random() < matchProbability;
};

/**
 * Get a single match by ID
 */
export const getMatchById = async (matchId: string, matches: Match[]): Promise<Match | null> => {
  const match = matches.find(m => m.id === matchId);
  return match || null;
};

/**
 * Sort matches by last message timestamp or creation date
 */
export const sortMatchesByRecency = (matches: Match[]): Match[] => {
  return [...matches].sort((a, b) => {
    const aTime = a.lastMessageTimestamp || a.createdAt;
    const bTime = b.lastMessageTimestamp || b.createdAt;
    
    // Convert to milliseconds for comparison
    const aMs = aTime.toMillis ? aTime.toMillis() : (aTime as any).getTime();
    const bMs = bTime.toMillis ? bTime.toMillis() : (bTime as any).getTime();
    
    return bMs - aMs; // Most recent first
  });
}; 