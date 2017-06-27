# FacebookLearner
Character prediction RNN thingo that learns to emulate a group chat hopefully maybe

## Input
The program expects `messages.json` to be a file that conforms to the following spec.
+ `letters` is a string containing all characters present within messages (to build the input vector to the network)
+ `authors` is an array containing a unique identifier for each group chat participant
+ `messages` is an array containing message objects.
+ Each message object contains an `author`, which must be present in `authors`, and also a `message`,
     which can only be    comprised of characters in `letters`. 
```json
{
  "letters": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
  "authors": [
    {"email": "some@unique.string", "name": "alice"},
    {"email": "bobisthebest@gmail.com", "name": "bob"}
  ],
  "messages": [
    {
      "email": "some@unique.string",
      "message": "Hi bob here s my public key"
    },
    {
      "email": "bobisthebest@gmail.com",
      "message": "fuck you alice im sick of your shit"
    },
    {
      "email": "some@unique.string",
      "message": "REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
    }
  ]
}
```
