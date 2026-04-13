import { User } from '../models/User';
import { GroupChat, GroupMessage } from '../models/Group';
import { Timestamp } from 'firebase/firestore';
import { getAllUsers } from './sampleDataService';

// Current user ID for testing - in a real app this would come from authentication
const CURRENT_USER_ID = '00001';

/**
 * Generate sample group chats for development
 */
export const generateSampleGroups = async (count: number = 3): Promise<GroupChat[]> => {
  try {
    // Get all users for members
    const allUsers = await getAllUsers();
    
    // Sample group data
    const sampleGroups: Partial<GroupChat>[] = [
      {
        name: "Chicago Downtown Apartments",
        description: "Has anyone toured the units on Michigan Ave?",
        tags: ["Downtown", "Michigan Ave", "Luxury"],
        isPublic: true,
      },
      {
        name: "Budget Friendly Options",
        description: "I found a great deal in Wicker Park",
        tags: ["Budget", "Wicker Park", "Deals"],
        isPublic: true,
      },
      {
        name: "Lincoln Park Roommates",
        description: "Looking for 2 more people to join our apartment",
        tags: ["Lincoln Park", "Roommates", "Available Now"],
        isPublic: true,
      },
      {
        name: "West Loop Apartment Share",
        description: "3BR in West Loop, need 1 more roommate",
        tags: ["West Loop", "Roommates", "3BR"],
        isPublic: true,
      },
      {
        name: "UChicago Housing",
        description: "For students and staff looking near campus",
        tags: ["UChicago", "Hyde Park", "Students"],
        isPublic: true,
      }
    ];
    
    // Create complete group objects
    const groups: GroupChat[] = sampleGroups.slice(0, count).map((group, index) => {
      // Randomly select 3-8 members for each group
      const memberCount = Math.floor(Math.random() * 6) + 3; // 3-8 members
      const shuffled = [...allUsers].sort(() => 0.5 - Math.random());
      const randomMembers = shuffled.slice(0, memberCount);
      
      // Ensure current user is a member of all groups
      if (!randomMembers.some(u => u.userId === CURRENT_USER_ID)) {
        randomMembers[0] = allUsers.find(u => u.userId === CURRENT_USER_ID) || randomMembers[0];
      }
      
      const memberIds = randomMembers.map(user => user.userId || user.id);
      
      // Create a random timestamp within the last week
      const daysAgo = Math.floor(Math.random() * 7); // 0-6 days ago
      const hoursAgo = Math.floor(Math.random() * 24); // 0-23 hours ago
      const createdAt = new Date();
      createdAt.setDate(createdAt.getDate() - daysAgo);
      createdAt.setHours(createdAt.getHours() - hoursAgo);
      
      // 50% chance of having a last message
      const hasLastMessage = Math.random() > 0.5;
      const lastMessageTimestamp = hasLastMessage ? new Date(createdAt.getTime() + Math.random() * 86400000) : undefined;
      const randomMember = randomMembers[Math.floor(Math.random() * randomMembers.length)];
      
      return {
        id: `group_${index + 1}`,
        name: group.name!,
        description: group.description!,
        createdBy: memberIds[0],
        createdAt: Timestamp.fromDate(createdAt),
        updatedAt: Timestamp.fromDate(lastMessageTimestamp || createdAt),
        lastMessageTimestamp: hasLastMessage ? Timestamp.fromDate(lastMessageTimestamp!) : undefined,
        lastMessageText: hasLastMessage ? getSampleMessage() : undefined,
        lastMessageSender: hasLastMessage ? randomMember.name : undefined,
        members: memberIds,
        memberCount: memberIds.length,
        groupImageUrl: getGroupImage(index),
        tags: group.tags || [],
        isPublic: group.isPublic || false,
        hasUnreadMessages: hasLastMessage ? { [CURRENT_USER_ID]: Math.random() > 0.5 } : undefined
      };
    });
    
    return groups;
  } catch (error) {
    console.error('Error generating sample groups:', error);
    throw error;
  }
};

/**
 * Get members of a group
 */
export const getGroupMembers = async (group: GroupChat): Promise<User[]> => {
  try {
    const allUsers = await getAllUsers();
    const groupMembers = allUsers.filter(user => 
      group.members.includes(user.userId || user.id)
    );
    return groupMembers;
  } catch (error) {
    console.error('Error getting group members:', error);
    throw error;
  }
};

/**
 * Generate random messages for a group chat
 */
export const generateGroupMessages = async (groupId: string, count: number = 10): Promise<GroupMessage[]> => {
  try {
    const group = (await generateSampleGroups()).find(g => g.id === groupId);
    if (!group) throw new Error(`Group with ID ${groupId} not found`);
    
    const members = await getGroupMembers(group);
    
    const messages: GroupMessage[] = [];
    const now = new Date();
    
    for (let i = 0; i < count; i++) {
      // Random sender from group members
      const sender = members[Math.floor(Math.random() * members.length)];
      
      // Random time in the past 24 hours, in chronological order
      const minutesAgo = (count - i) * 15 + Math.floor(Math.random() * 30);
      const timestamp = new Date(now.getTime() - minutesAgo * 60000);
      
      // Random message
      const message: GroupMessage = {
        id: `msg_${groupId}_${i}`,
        groupId,
        senderId: sender.userId || sender.id,
        senderName: sender.name,
        content: getSampleMessage(),
        timestamp: Timestamp.fromDate(timestamp),
        readBy: members.filter(() => Math.random() > 0.3).map(m => m.userId || m.id) // Random read status
      };
      
      messages.push(message);
    }
    
    return messages;
  } catch (error) {
    console.error('Error generating group messages:', error);
    throw error;
  }
};

/**
 * Get sample message content
 */
const getSampleMessage = (): string => {
  const messages = [
    "Has anyone seen the apartment on Michigan Ave?",
    "The rent seems too good to be true, any catches?",
    "I'm available to tour this weekend if anyone wants to join",
    "Just visited yesterday, the place is smaller than the photos show",
    "The location is perfect though, right by the L station",
    "Any recommendations for moving companies?",
    "Does the building allow pets?",
    "I'm looking for a place starting next month",
    "How's the neighborhood at night?",
    "Found a 3BR for $2100, should we check it out?",
    "I'm interested in joining if you still need someone",
    "What's the security deposit situation?",
    "Is parking included or extra?",
    "The amenities in that building are amazing",
    "Anyone want to share an Uber to the open house tomorrow?"
  ];
  
  return messages[Math.floor(Math.random() * messages.length)];
};

/**
 * Get a placeholder group image
 */
const getGroupImage = (index: number): string => {
  const colors = ['4a90e2', '27ae60', '8e44ad', 'e67e22', 'e74c3c', '2c3e50'];
  const color = colors[index % colors.length];
  return `https://via.placeholder.com/100/${color}/ffffff?text=Group`;
};

/**
 * Sort groups by activity (last message timestamp)
 */
export const sortGroupsByActivity = (groups: GroupChat[]): GroupChat[] => {
  return [...groups].sort((a, b) => {
    const aTime = a.lastMessageTimestamp || a.createdAt;
    const bTime = b.lastMessageTimestamp || b.createdAt;
    
    // Convert to milliseconds for comparison
    const aMs = aTime.toMillis ? aTime.toMillis() : (aTime as any).getTime();
    const bMs = bTime.toMillis ? bTime.toMillis() : (bTime as any).getTime();
    
    return bMs - aMs; // Most recent first
  });
};

/**
 * Get suggested groups based on user preferences
 */
export const getSuggestedGroups = async (allGroups: GroupChat[]): Promise<GroupChat[]> => {
  // In a real app, you would filter based on user preferences
  // For now, just return public groups the user is not a member of
  return allGroups.filter(group => 
    group.isPublic && !group.members.includes(CURRENT_USER_ID)
  );
};

/**
 * Join a group
 */
export const joinGroup = async (groupId: string, groups: GroupChat[]): Promise<GroupChat[]> => {
  return groups.map(group => {
    if (group.id === groupId && !group.members.includes(CURRENT_USER_ID)) {
      return {
        ...group,
        members: [...group.members, CURRENT_USER_ID],
        memberCount: group.memberCount + 1
      };
    }
    return group;
  });
};

/**
 * Create a new group
 */
export const createNewGroup = async (
  name: string, 
  description: string, 
  initialMembers: string[] = [],
  isPublic: boolean = true,
  tags: string[] = []
): Promise<GroupChat> => {
  // In a real app, this would create a document in Firestore
  const now = Timestamp.now();
  
  // Ensure creator is included in members
  if (!initialMembers.includes(CURRENT_USER_ID)) {
    initialMembers.push(CURRENT_USER_ID);
  }
  
  const newGroup: GroupChat = {
    id: `group_new_${now.toMillis()}`,
    name,
    description,
    createdBy: CURRENT_USER_ID,
    createdAt: now,
    updatedAt: now,
    members: initialMembers,
    memberCount: initialMembers.length,
    tags,
    isPublic,
    groupImageUrl: getGroupImage(Math.floor(Math.random() * 6)) // Random color
  };
  
  return newGroup;
}; 