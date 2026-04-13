import { 
  collection, 
  query, 
  where, 
  getDocs, 
  getDoc, 
  doc, 
  updateDoc, 
  arrayUnion, 
  arrayRemove,
  limit,
  startAfter,
  orderBy,
  Timestamp,
  setDoc
} from 'firebase/firestore';
import { getDownloadURL, ref, uploadBytes } from 'firebase/storage';
import { db, storage } from '../config/firebaseConfig';
import UserProfile from '../models/UserProfile';
import Message, { Conversation } from '../models/Message';
import authService from './AuthService';

class UserService {
  // Get user profile by ID
  async getUserById(userId) {
    try {
      const userRef = doc(db, 'users', userId);
      const userSnap = await getDoc(userRef);
      
      if (userSnap.exists()) {
        return new UserProfile(userSnap.data());
      } else {
        throw new Error('User not found');
      }
    } catch (error) {
      console.error('Error getting user:', error);
      throw error;
    }
  }
  
  // Get potential roommates for swiping
  async getPotentialRoommates(lastVisibleUser = null, batchSize = 10) {
    try {
      const currentUser = await authService.getCurrentUserProfile();
      
      if (!currentUser) {
        throw new Error('No authenticated user');
      }
      
      let q;
      
      // Create base query
      if (lastVisibleUser) {
        q = query(
          collection(db, 'users'),
          where('accountType', 'in', ['roommate', 'has_room']),
          where('id', 'not-in', [...currentUser.likes, ...currentUser.dislikes, currentUser.id]),
          orderBy('id'),
          startAfter(lastVisibleUser),
          limit(batchSize)
        );
      } else {
        q = query(
          collection(db, 'users'),
          where('accountType', 'in', ['roommate', 'has_room']),
          where('id', 'not-in', [...currentUser.likes, ...currentUser.dislikes, currentUser.id]),
          orderBy('id'),
          limit(batchSize)
        );
      }
      
      const querySnapshot = await getDocs(q);
      const roommates = [];
      let lastVisible = null;
      
      querySnapshot.forEach(doc => {
        roommates.push(new UserProfile(doc.data()));
        lastVisible = doc;
      });
      
      return { roommates, lastVisible };
    } catch (error) {
      console.error('Error getting potential roommates:', error);
      throw error;
    }
  }
  
  // Like a user
  async likeUser(likedUserId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      // Update current user's likes
      const currentUserRef = doc(db, 'users', currentUserId);
      await updateDoc(currentUserRef, {
        likes: arrayUnion(likedUserId)
      });
      
      // Check if the other user has already liked current user (mutual match)
      const likedUserRef = doc(db, 'users', likedUserId);
      const likedUserSnap = await getDoc(likedUserRef);
      
      if (likedUserSnap.exists()) {
        const likedUserData = likedUserSnap.data();
        
        if (likedUserData.likes && likedUserData.likes.includes(currentUserId)) {
          // It's a match! Add to both users' matches
          await updateDoc(currentUserRef, {
            matches: arrayUnion(likedUserId)
          });
          
          await updateDoc(likedUserRef, {
            matches: arrayUnion(currentUserId)
          });
          
          // Create a conversation between the users
          const conversationId = [currentUserId, likedUserId].sort().join('_');
          const conversationRef = doc(db, 'conversations', conversationId);
          
          const newConversation = new Conversation({
            id: conversationId,
            participants: [currentUserId, likedUserId],
            lastActivity: new Date(),
            createdAt: new Date(),
            unreadCount: {
              [currentUserId]: 0,
              [likedUserId]: 0
            }
          });
          
          await setDoc(conversationRef, newConversation.toJSON());
          
          return { isMatch: true, matchedUserId: likedUserId };
        }
      }
      
      return { isMatch: false };
    } catch (error) {
      console.error('Error liking user:', error);
      throw error;
    }
  }
  
  // Dislike a user
  async dislikeUser(dislikedUserId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const currentUserRef = doc(db, 'users', currentUserId);
      await updateDoc(currentUserRef, {
        dislikes: arrayUnion(dislikedUserId)
      });
      
      return true;
    } catch (error) {
      console.error('Error disliking user:', error);
      throw error;
    }
  }
  
  // Get matched profiles
  async getMatches() {
    try {
      const currentUser = await authService.getCurrentUserProfile();
      
      if (!currentUser || !currentUser.matches) {
        return [];
      }
      
      const matches = [];
      
      for (const matchId of currentUser.matches) {
        try {
          const matchProfile = await this.getUserById(matchId);
          matches.push(matchProfile);
        } catch (error) {
          console.error(`Error getting match ${matchId}:`, error);
        }
      }
      
      return matches;
    } catch (error) {
      console.error('Error getting matches:', error);
      throw error;
    }
  }
  
  // Get liked profiles
  async getLikedProfiles() {
    try {
      const currentUser = await authService.getCurrentUserProfile();
      
      if (!currentUser || !currentUser.likes) {
        return [];
      }
      
      const liked = [];
      
      for (const likedId of currentUser.likes) {
        try {
          const likedProfile = await this.getUserById(likedId);
          liked.push(likedProfile);
        } catch (error) {
          console.error(`Error getting liked profile ${likedId}:`, error);
        }
      }
      
      return liked;
    } catch (error) {
      console.error('Error getting liked profiles:', error);
      throw error;
    }
  }
  
  // Upload profile image
  async uploadProfileImage(uri, fileName) {
    try {
      const response = await fetch(uri);
      const blob = await response.blob();
      
      const storageRef = ref(storage, `profileImages/${fileName}`);
      await uploadBytes(storageRef, blob);
      
      const downloadURL = await getDownloadURL(storageRef);
      return downloadURL;
    } catch (error) {
      console.error('Error uploading profile image:', error);
      throw error;
    }
  }
  
  // Get conversations
  async getConversations() {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const q = query(
        collection(db, 'conversations'),
        where('participants', 'array-contains', currentUserId),
        orderBy('lastActivity', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      const conversations = [];
      
      for (const doc of querySnapshot.docs) {
        const conversation = new Conversation(doc.data());
        
        // Get other participant's info
        const otherParticipantId = conversation.participants.find(id => id !== currentUserId);
        if (otherParticipantId) {
          try {
            const otherUser = await this.getUserById(otherParticipantId);
            conversation.name = otherUser.fullName;
            conversation.imageUrl = otherUser.profileImageUrl;
          } catch (error) {
            console.error(`Error getting conversation participant ${otherParticipantId}:`, error);
          }
        }
        
        conversations.push(conversation);
      }
      
      return conversations;
    } catch (error) {
      console.error('Error getting conversations:', error);
      throw error;
    }
  }
  
  // Get messages for a conversation
  async getMessages(conversationId, lastMessageTimestamp = null, limit = 20) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      let q;
      
      if (lastMessageTimestamp) {
        q = query(
          collection(db, 'conversations', conversationId, 'messages'),
          orderBy('timestamp', 'desc'),
          startAfter(new Timestamp(lastMessageTimestamp.seconds, lastMessageTimestamp.nanoseconds)),
          limit(limit)
        );
      } else {
        q = query(
          collection(db, 'conversations', conversationId, 'messages'),
          orderBy('timestamp', 'desc'),
          limit(limit)
        );
      }
      
      const querySnapshot = await getDocs(q);
      const messages = [];
      let lastVisible = null;
      
      querySnapshot.forEach(doc => {
        messages.push(new Message(doc.data()));
        lastVisible = doc.data().timestamp;
      });
      
      // Mark messages as read
      const conversationRef = doc(db, 'conversations', conversationId);
      await updateDoc(conversationRef, {
        [`unreadCount.${currentUserId}`]: 0
      });
      
      return { messages: messages.reverse(), lastVisible };
    } catch (error) {
      console.error('Error getting messages:', error);
      throw error;
    }
  }
  
  // Send a message
  async sendMessage(conversationId, receiverId, messageData) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const newMessage = new Message({
        ...messageData,
        conversationId,
        senderId: currentUserId,
        receiverId,
        timestamp: new Date(),
      });
      
      // Add message to conversation
      const messageRef = doc(collection(db, 'conversations', conversationId, 'messages'));
      newMessage.id = messageRef.id;
      
      await setDoc(messageRef, newMessage.toJSON());
      
      // Update conversation's last message and activity
      const conversationRef = doc(db, 'conversations', conversationId);
      
      await updateDoc(conversationRef, {
        lastMessage: {
          text: newMessage.text,
          senderId: currentUserId,
          timestamp: newMessage.timestamp
        },
        lastActivity: newMessage.timestamp,
        [`unreadCount.${receiverId}`]: arrayUnion(1)
      });
      
      return newMessage;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }
}

export default new UserService(); 