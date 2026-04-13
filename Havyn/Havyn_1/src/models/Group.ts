import { Timestamp } from 'firebase/firestore';

export interface GroupChat {
  id: string;
  name: string;
  description: string;
  createdBy: string; // userId of creator
  createdAt: Timestamp;
  updatedAt: Timestamp;
  lastMessageTimestamp?: Timestamp;
  lastMessageText?: string;
  lastMessageSender?: string;
  members: string[]; // array of userIds
  memberCount: number;
  groupImageUrl?: string;
  tags?: string[]; // for filtering (e.g., "Budget Friendly", "Downtown", "Lincoln Park")
  isPublic: boolean; // if true, appears in suggestions; if false, only visible to members
  hasUnreadMessages?: {[userId: string]: boolean}; // track unread status per user
}

export interface GroupMessage {
  id: string;
  groupId: string;
  senderId: string;
  senderName: string;
  content: string;
  timestamp: Timestamp;
  readBy: string[]; // array of userIds who have read the message
} 