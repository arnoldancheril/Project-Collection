import { Timestamp } from 'firebase/firestore';

export type MessageType = 'text' | 'image' | 'system';

export interface Message {
  id: string;
  matchId: string; // Reference to the match between users
  senderId: string;
  receiverId: string;
  content: string;
  type: MessageType;
  createdAt: Timestamp;
  read: boolean;
} 