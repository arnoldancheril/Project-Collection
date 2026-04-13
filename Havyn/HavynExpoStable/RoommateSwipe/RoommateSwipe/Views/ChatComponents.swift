//
//  ChatComponents.swift
//  RoommateSwipe
//
//  Created by  on 3/25/25.
//

import SwiftUI

// Chat Message Model
struct ChatMessage: Identifiable {
    var id: String
    var sender: String
    var text: String
    var time: String
    var isCurrentUser: Bool
}

// Chat Bubble Component
struct ChatBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isCurrentUser { Spacer() }
            
            VStack(alignment: message.isCurrentUser ? .trailing : .leading, spacing: 4) {
                if !message.isCurrentUser {
                    Text(message.sender)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Text(message.text)
                    .padding(12)
                    .background(message.isCurrentUser ? Color.blue : Color(.secondarySystemBackground))
                    .foregroundColor(message.isCurrentUser ? .white : .primary)
                    .cornerRadius(16)
                
                Text(message.time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if !message.isCurrentUser { Spacer() }
        }
    }
} 