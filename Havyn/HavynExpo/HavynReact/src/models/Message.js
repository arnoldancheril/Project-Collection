export default class Message {
  constructor(data = {}) {
    this.id = data.id || '';
    this.conversationId = data.conversationId || '';
    this.senderId = data.senderId || '';
    this.receiverId = data.receiverId || '';
    this.text = data.text || '';
    this.imageUrl = data.imageUrl || '';
    this.timestamp = data.timestamp || new Date();
    this.isRead = data.isRead || false;
    this.replyToMessageId = data.replyToMessageId || null;
    this.type = data.type || 'text'; // text, image, location, audio, video
    this.locationData = data.locationData || null;
    this.status = data.status || 'sent'; // sent, delivered, read
  }
  
  toJSON() {
    return {
      id: this.id,
      conversationId: this.conversationId,
      senderId: this.senderId,
      receiverId: this.receiverId,
      text: this.text,
      imageUrl: this.imageUrl,
      timestamp: this.timestamp,
      isRead: this.isRead,
      replyToMessageId: this.replyToMessageId,
      type: this.type,
      locationData: this.locationData,
      status: this.status,
    };
  }
}

export class Conversation {
  constructor(data = {}) {
    this.id = data.id || '';
    this.participants = data.participants || []; // Array of user IDs
    this.lastMessage = data.lastMessage || null;
    this.lastActivity = data.lastActivity || new Date();
    this.createdAt = data.createdAt || new Date();
    this.isGroupChat = data.isGroupChat || false;
    this.name = data.name || ''; // For group chats
    this.imageUrl = data.imageUrl || ''; // For group chats
    this.unreadCount = data.unreadCount || {};
  }
  
  toJSON() {
    return {
      id: this.id,
      participants: this.participants,
      lastMessage: this.lastMessage,
      lastActivity: this.lastActivity,
      createdAt: this.createdAt,
      isGroupChat: this.isGroupChat,
      name: this.name,
      imageUrl: this.imageUrl,
      unreadCount: this.unreadCount,
    };
  }
} 