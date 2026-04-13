import { Timestamp } from 'firebase/firestore';

export type MatchStatus = 'pending' | 'accepted' | 'rejected' | 'archived';

export interface Match {
  id: string;
  initiatorId: string; // User who sent the match request
  recipientId: string; // User who received the match request
  listingId?: string; // Optional, only if match is related to a specific listing
  status: MatchStatus;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  lastMessageTimestamp?: Timestamp;
  hasUnreadMessages?: boolean;
} 