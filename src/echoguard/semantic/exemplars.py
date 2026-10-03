"""Python scam and benign exemplars used by the MiniLM similarity scorer.

Android loads its bundled embedding JSON rather than importing these lists.
"""

SCAM_EXEMPLARS = [
    "Your account will be blocked if you do not act immediately.",
    "Please share the OTP that was just sent to your phone.",
    "This is an urgent call from your bank's security department.",
    "Do not tell anyone about this call, it is confidential.",
    "We need to verify your card details right now.",
    "Install this app so I can access your screen remotely.",
    "There is a legal case against you, pay now to resolve it.",
    "Your KYC has expired, share your Aadhaar and bank details.",
    "You have won a prize, just pay a small processing fee first.",
    "This is the income tax department, you must pay immediately or face arrest.",
    "I have kidnapped your son, if you want him back give me some amount.",
    "Your child met with an accident and is admitted in the hospital, send money for treatment.",
    "Congratulations, you have won a KBC lottery of 25 lakhs. Pay the tax amount to claim it.",
    "Your electricity bill is pending, power will be disconnected at 9 PM tonight.",
    "You have received a cashback reward, click the link to claim it into your account.",
    "This is Microsoft technical support, your computer has a virus.",
]

BENIGN_EXEMPLARS = [
    "Hi mom, I am running late, save some dinner for me.",
    "Could you please tell me my account balance?",
    "I need to book a flight to Mumbai for tomorrow evening.",
    "Yeah that sounds good, let's meet at the coffee shop around 4 PM.",
    "I kidnapped your child, give me some amount. Just kidding, I was joking.",
    "Share your OTP and bank details. Just kidding, it's a prank.",
    "I am calling from the police, you will be arrested. Relax, I am just kidding.",
    "Hello, I kidnaped your child. Okay, I was kidding. But I said I was kidding",
    "Hi, just calling to check how you're doing.",
    "Can we reschedule our meeting to next week?",
    "Your order has been shipped and will arrive on Friday.",
    "This is a reminder about your appointment tomorrow.",
    "Hello, this is your bank's security department calling to confirm a transaction you made today, no action is needed if this was you.",
    "This is a courtesy call from customer service, we noticed unusual activity and wanted to verify it was really you, can you confirm the last transaction?",
    "Good afternoon, this is the telecom support team, we're calling to let you know your plan renewal is due, would you like to upgrade?",
    "Hi, this is your bank, we're just verifying your recent address change, no OTP or payment is needed for this call.",
    "This is a reminder that your card is expiring soon, you can request a replacement through the app, no need to share any details on this call.",
]
